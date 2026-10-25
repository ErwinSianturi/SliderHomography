#!/usr/bin/env python3
"""Stage 1 (DeepStream container on host GPU 1 via --gpus device=1).

Runs the production person pgie (rendered shard config, copied as pgie_production_shard.txt; gpu-id=0 is the
only visible GPU inside the container) on each still frame through production muxer settings
(nvstreammux 960x544, enable-padding=0, batch-size=8, gpu-id=0). Writes detections in muxer coordinates
(960x544) to /out/det/<frame>.json. Projection happens in stage 2 on the host.
"""
import glob
import json
import os

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

try:
    import pyds  # noqa: E402
except ImportError:
    # Same install as run_gst_pipeline.py _install_pyds (--no-deps wheel from models/), but into a target dir
    # so the container's site-packages is not modified.
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", "/tmp/pyds_site",
                    "/workspace/models/pyds-1.2.3-cp312-cp312-linux_x86_64.whl"], check=True)
    sys.path.insert(0, "/tmp/pyds_site")
    import pyds  # noqa: E402

FRAMES = "/out/frames"
DET = "/out/det"
CFG = "/out/pgie_production_shard.txt"
MUX_W, MUX_H = 960, 544

Gst.init(None)
os.makedirs(DET, exist_ok=True)
frames = sorted(glob.glob(os.path.join(FRAMES, "*.jpg")))
print("frames", len(frames), flush=True)


def make_pipeline(path):
    desc = (
        f"filesrc location={path} ! jpegdec ! nvvideoconvert ! "
        "video/x-raw(memory:NVMM),format=NV12 ! mux.sink_0 "
        f"nvstreammux name=mux batch-size=8 width={MUX_W} height={MUX_H} live-source=0 "
        "buffer-pool-size=4 batched-push-timeout=10000 enable-padding=0 nvbuf-memory-type=0 "
        "gpu-id=0 sync-inputs=0 max-latency=0 ! "
        f"nvinfer name=pgie config-file-path={CFG} ! fakesink sync=false"
    )
    return Gst.parse_launch(desc)


def on_src_buffer(pad, info, store):
    buf = info.get_buffer()
    if buf is None:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if batch_meta is None:
        return Gst.PadProbeReturn.OK
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            obj = pyds.NvDsObjectMeta.cast(l_obj.data)
            r = obj.rect_params
            store.append({"class_id": int(obj.class_id), "conf": float(obj.confidence),
                          "left": float(r.left), "top": float(r.top),
                          "width": float(r.width), "height": float(r.height)})
            l_obj = l_obj.next
        l_frame = l_frame.next
    return Gst.PadProbeReturn.OK


total = 0
errors_total = 0
for path in frames:
    name = os.path.basename(path)
    store = []
    errors = []
    pipe = make_pipeline(path)
    pgie = pipe.get_by_name("pgie")
    pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, on_src_buffer, store)
    loop = GLib.MainLoop()
    bus = pipe.get_bus()
    bus.add_signal_watch()

    def on_message(bus, msg, loop=loop, errors=errors):
        if msg.type == Gst.MessageType.EOS:
            loop.quit()
        elif msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            errors.append(f"{err}: {dbg}")
            loop.quit()

    bus.connect("message", on_message)
    pipe.set_state(Gst.State.PLAYING)
    loop.run()
    pipe.set_state(Gst.State.NULL)
    bus.remove_signal_watch()
    out = {"file": name, "mux": [MUX_W, MUX_H], "config": os.path.basename(CFG),
           "errors": errors, "detections": store}
    with open(os.path.join(DET, name[:-4] + ".json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    total += len(store)
    errors_total += len(errors)
    print(name, "dets", len(store), "errors", len(errors), flush=True)
print("done frames", len(frames), "dets", total, "errors", errors_total, flush=True)
