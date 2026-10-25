#!/usr/bin/env python3
"""Stage 2 (repo .venv on RTX, production code).

Stage-1 detections (muxer space 960x544) -> native frame px -> foot point (src.geometry.homography.foot,
bottom_center as in config.yaml) -> production Projector built from the Downloads export -> map px.
Writes detections.csv / detections.json, one overlay PNG per camera, and summary.json.
"""
import collections
import csv
import glob
import hashlib
import json
import math
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

REPO = "/home/inaai-server-1/inaai/multi-camera-people-tracking-product-interaction-deepstream"
SCR = REPO + "/.workspace/detect_still_20261024"
EXPORT = SCR + "/cameras_export_downloads.json"
FRAMES = SCR + "/frames"
DET = SCR + "/det"
OUT = SCR + "/out"
FLOOR = REPO + "/configs/geometry/floorplan.png"
MUX_W, MUX_H = 960, 544
FOOT_MODE = "bottom_center"
FOOT_RATIO = 0.92
MARGIN = 100
GUTTER = 40
HEADER = 64
R = 6
VALID = (0, 150, 0)
FLAGGED = (230, 110, 0)

os.chdir(REPO)
sys.path.insert(0, REPO)
from src.geometry.homography import foot  # noqa: E402
from src.geometry.projector import Projector  # noqa: E402

font_h = ImageFont.load_default(size=20)
font_l = ImageFont.load_default(size=16)


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def sha12(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


dl = json.load(open(EXPORT, encoding="utf-8"))
homog = {h["cameraId"]: h for h in dl["homography"]}
cams = list(homog)
refs = collections.defaultdict(set)
for key in ("frameZones", "frameRegions", "frameCrops", "frameCountingLines", "frameFlowArrows"):
    for it in dl.get(key) or []:
        if it.get("frameImage") and it.get("cameraId"):
            refs[it["cameraId"]].add(it["frameImage"].split("/")[-1])
frame_cam = collections.defaultdict(set)
for c in cams:
    for n in refs.get(c, ()):
        if n.endswith("_1.jpg"):
            frame_cam[n].add(c)
frames = sorted(os.path.basename(p) for p in glob.glob(FRAMES + "/*_1.jpg"))
for base in frames:
    prefix = base[: -len("_1.jpg")]
    if base not in frame_cam and prefix in homog:
        frame_cam[base].add(prefix)
frame_map = {}
for base in frames:
    cs = frame_cam.get(base, set())
    if len(cs) != 1:
        raise SystemExit(f"frame {base} maps to {sorted(cs)}")
    frame_map[base] = next(iter(cs))
if len(set(frame_map.values())) != len(frame_map):
    raise SystemExit("frame to camera mapping is not one-to-one")
print("frames", len(frame_map), "cameras", len(set(frame_map.values())), flush=True)

projs = {c: Projector(c, export=dl) for c in cams}
# Self-check: the production projector should reproduce the export's own frame->map pairs.
medians, worst = [], []
for c in cams:
    errs = []
    for p in homog[c]["pairs"]:
        if p.get("framePt") is None or p.get("mapPt") is None:
            continue
        m = projs[c].to_map(tuple(p["framePt"]), validate=False)
        errs.append(float("inf") if m is None else math.hypot(m[0] - p["mapPt"][0], m[1] - p["mapPt"][1]))
    errs.sort()
    if errs:
        medians.append(errs[len(errs) // 2])
        worst.append((errs[-1], c))
medians.sort()
print("projector self-check: median of per-camera median px", round(medians[len(medians) // 2], 2),
      "| worst camera", max(worst), flush=True)

floor = Image.open(FLOOR).convert("RGB")
FW, FH = floor.size
print("floorplan", FW, FH, "sha", sha12(FLOOR), flush=True)
os.makedirs(OUT + "/images", exist_ok=True)

rows, per_frame = [], {}
for base, cam in frame_map.items():
    d = json.load(open(os.path.join(DET, base[:-4] + ".json"), encoding="utf-8"))
    if d["errors"]:
        raise SystemExit(f"stage1 error for {base}: {d['errors']}")
    W, H = Image.open(os.path.join(FRAMES, base)).size
    sx, sy = W / MUX_W, H / MUX_H
    items = []
    for n, det in enumerate(d["detections"], start=1):
        l = det["left"] * sx
        t = det["top"] * sy
        r = (det["left"] + det["width"]) * sx
        b = (det["top"] + det["height"]) * sy
        fx, fy = foot(l, t, r, b, foot_mode=FOOT_MODE, foot_ratio=FOOT_RATIO)
        m = projs[cam].to_map((fx, fy), validate=False)
        ok = projs[cam].to_map((fx, fy), validate=True) is not None
        items.append({
            "frame": base, "camera": cam, "n": n, "class_id": det["class_id"], "conf": round(det["conf"], 4),
            "bbox_native_ltrb": [round(l, 1), round(t, 1), round(r, 1), round(b, 1)],
            "foot_native": [round(fx, 1), round(fy, 1)],
            "map_px": None if m is None else [round(m[0], 1), round(m[1], 1)],
            "map_valid_production": ok,
        })
    per_frame[base] = {"camera": cam, "native": [W, H], "items": items}
    rows.extend(items)

with open(OUT + "/detections.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)
fields = ["frame", "camera", "n", "class_id", "conf", "bbox_native_ltrb", "foot_native", "map_px", "map_valid_production"]
with open(OUT + "/detections.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for row in rows:
        w.writerow({k: (json.dumps(row[k]) if isinstance(row[k], list) else row[k]) for k in fields})

for base, info in per_frame.items():
    cam, (W, H), items = info["camera"], info["native"], info["items"]
    frame = Image.open(os.path.join(FRAMES, base)).convert("RGB")
    mapped = [it["map_px"] for it in items if it["map_px"] is not None]
    crop, x0, y0 = None, 0, 0
    if mapped:
        xs = [p[0] for p in mapped]
        ys = [p[1] for p in mapped]
        x0 = max(0, math.floor(min(xs) - MARGIN))
        y0 = max(0, math.floor(min(ys) - MARGIN))
        x1 = min(FW, math.ceil(max(xs) + MARGIN))
        y1 = min(FH, math.ceil(max(ys) + MARGIN))
        crop = floor.crop((x0, y0, x1, y1))
    cw, ch = crop.size if crop is not None else (420, 240)
    nvalid = sum(1 for it in items if it["map_valid_production"])
    line1 = f"{cam} | frame {base} {W}x{H} | deteksi {len(items)} | lolos validasi produksi {nvalid}, ditandai {len(items) - nvalid}"
    if crop is not None:
        line2 = f"detector NEW_0 int8 | kaki bottom_center | homography Downloads | hijau lolos, oranye ditandai | map x {x0}-{x1}, y {y0}-{y1}"
    else:
        line2 = "detector NEW_0 int8 | homography Downloads | tidak ada titik map untuk kamera ini"
    canvas_w = max(W + GUTTER + cw, int(font_h.getlength(line1)) + 20, int(font_h.getlength(line2)) + 20)
    canvas = Image.new("RGB", (canvas_w, HEADER + max(H, ch)), (255, 255, 255))
    canvas.paste(frame, (0, HEADER))
    if crop is not None:
        canvas.paste(crop, (W + GUTTER, HEADER))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 6), line1, fill=(0, 0, 0), font=font_h)
    draw.text((10, 32), line2, fill=(0, 0, 0), font=font_h)
    for it in items:
        color = VALID if it["map_valid_production"] else FLAGGED
        l, t, r, b = it["bbox_native_ltrb"]
        draw.rectangle([l, t + HEADER, r, b + HEADER], outline=color, width=2)
        fx, fy = it["foot_native"]
        draw.ellipse([fx - R, fy + HEADER - R, fx + R, fy + HEADER + R], fill=color, outline=(255, 255, 255))
        draw.text((l + 3, t + HEADER + 2), str(it["n"]), fill=color, font=font_l,
                  stroke_width=2, stroke_fill=(255, 255, 255))
        if it["map_px"] is not None and crop is not None:
            mx, my = it["map_px"]
            px = W + GUTTER + (mx - x0)
            py = HEADER + (my - y0)
            draw.ellipse([px - R, py - R, px + R, py + R], fill=color, outline=(255, 255, 255))
            draw.text((px + R + 3, py - R - 3), str(it["n"]), fill=color, font=font_l,
                      stroke_width=2, stroke_fill=(255, 255, 255))
    canvas.save(os.path.join(OUT, "images", sanitize(cam) + ".png"))

summary = {
    "frames": len(per_frame),
    "detections": len(rows),
    "map_valid_production": sum(1 for r in rows if r["map_valid_production"]),
    "map_flagged_production": sum(1 for r in rows if not r["map_valid_production"]),
    "no_map_px": sum(1 for r in rows if r["map_px"] is None),
    "frames_without_detections": sorted(f for f, v in per_frame.items() if not v["items"]),
    "export_sha12": sha12(EXPORT),
    "floor_sha12": sha12(FLOOR),
}
with open(OUT + "/summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "frames_without_detections"}, ensure_ascii=False), flush=True)
print("frames without detections:", len(summary["frames_without_detections"]), flush=True)
