#!/usr/bin/env python3
"""Side-by-side check: export frame (native px, no scale) | floorplan crop, pairs numbered and joined by a line."""
import collections
import hashlib
import json
import math
import os
import re

from PIL import Image, ImageDraw, ImageFont

BASE = "/Users/hafidhsoekma/Projects/Pekerjaan/InaAI/MayoraShopperAnalytics/multi-camera-people-tracking-product-interaction-deepstream"
OUT = BASE + "/.workspace/.hand_off/analysis/homography_visual_downloads_20261023"
FLOOR = BASE + "/configs/geometry/floorplan.png"
EXPORT = OUT + "/cameras_export.json"
FRAMES_SRC = OUT + "/frames_src"
EXTRA_FRAMES = OUT + "/extra_src/frames"
EXTRA_THUMBS = OUT + "/extra_src/thumbs"
IMG_DIR = OUT + "/images"
FONT_PATH = "/System/Library/Fonts/Helvetica.ttc"
MARGIN = 100
GUTTER = 40
LINE_H = 26
HEADER = 2 * LINE_H + 12
R = 8
LINE_W = 3
PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40),
           (128, 0, 0), (0, 0, 128), (128, 128, 0), (255, 215, 180), (170, 255, 195),
           (128, 128, 128), (0, 0, 0), (220, 190, 255), (250, 190, 212), (255, 127, 0)]


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


dl = json.load(open(EXPORT, encoding="utf-8"))
refs = collections.defaultdict(set)
for key in ("frameZones", "frameRegions", "frameCrops", "frameCountingLines", "frameFlowArrows"):
    for it in dl.get(key) or []:
        if it.get("frameImage") and it.get("cameraId"):
            refs[it["cameraId"]].add(it["frameImage"])

jobs = []
for h in dl["homography"]:
    cam = h["cameraId"]
    names = sorted(refs.get(cam, set()))
    ones = [n for n in names if n.split("/")[-1].endswith("_1.jpg")]
    if ones:
        if len(ones) != 1:
            raise SystemExit(f"more than one _1 frame for {cam}: {ones}")
        base = ones[0].split("/")[-1]
        jobs.append((cam, base[:-4], os.path.join(FRAMES_SRC, base), base))
    else:
        if not names:
            raise SystemExit(f"no frame reference for {cam}")
        for n in names:
            base = n.split("/")[-1]
            if "/frames/" in n:
                src = os.path.join(EXTRA_FRAMES, base)
            elif "/thumbs/" in n:
                src = os.path.join(EXTRA_THUMBS, base)
            else:
                raise SystemExit(f"unknown frame folder: {n}")
            jobs.append((cam, base[:-4], src, base))
for cam, label, src, name in jobs:
    if not os.path.exists(src):
        raise SystemExit(f"missing source {src}")

floor = Image.open(FLOOR).convert("RGB")
FW, FH = floor.size
if (FW, FH) != (3750, 3368):
    raise SystemExit(f"unexpected floorplan size {FW}x{FH}")

font_header = ImageFont.truetype(FONT_PATH, 20, index=0)
font_label = ImageFont.truetype(FONT_PATH, 18, index=0)
os.makedirs(IMG_DIR, exist_ok=True)
pairs_by_cam = {h["cameraId"]: h["pairs"] for h in dl["homography"]}
manifest, problems, seen = [], [], set()

for cam, label, src, src_name in jobs:
    frame = Image.open(src).convert("RGB")
    W, H = frame.size
    drawn, skipped = [], []
    for p in pairs_by_cam[cam]:
        if p.get("framePt") is None or p.get("mapPt") is None:
            reason = "framePt null" if p.get("framePt") is None else "mapPt null"
            skipped.append({"id": p["id"], "reason": reason})
            continue
        drawn.append(p)
    if not drawn:
        problems.append((cam, "no drawable pairs"))
        continue
    oob = [p["id"] for p in drawn if not (0 <= p["framePt"][0] < W and 0 <= p["framePt"][1] < H)]
    if oob:
        problems.append((cam, label, "framePt outside frame", oob))
    mxs = [float(p["mapPt"][0]) for p in drawn]
    mys = [float(p["mapPt"][1]) for p in drawn]
    x0 = max(0, math.floor(min(mxs) - MARGIN))
    y0 = max(0, math.floor(min(mys) - MARGIN))
    x1 = min(FW, math.ceil(max(mxs) + MARGIN))
    y1 = min(FH, math.ceil(max(mys) + MARGIN))
    if x1 <= x0 or y1 <= y0:
        problems.append((cam, "empty crop"))
        continue
    crop = floor.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    line1 = f"{cam} | frame {src_name} {W}x{H} | pairs digambar {len(drawn)}, dilewati {len(skipped)}"
    line2 = f"floorplan crop x {x0}-{x1}, y {y0}-{y1} (mapPt + margin {MARGIN} px, piksel asli, tanpa scale)"
    canvas_w = max(W + GUTTER + cw, int(font_header.getlength(line1)) + 20, int(font_header.getlength(line2)) + 20)
    canvas_h = HEADER + max(H, ch)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    canvas.paste(frame, (0, HEADER))
    canvas.paste(crop, (W + GUTTER, HEADER))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 6), line1, fill=(0, 0, 0), font=font_header)
    draw.text((10, 6 + LINE_H), line2, fill=(0, 0, 0), font=font_header)
    pts = []
    for i, p in enumerate(drawn):
        fx, fy = float(p["framePt"][0]), float(p["framePt"][1])
        mx, my = float(p["mapPt"][0]), float(p["mapPt"][1])
        pts.append((p["id"], PALETTE[i % len(PALETTE)], (fx, fy + HEADER), (W + GUTTER + (mx - x0), HEADER + (my - y0))))
    for _, color, f_xy, m_xy in pts:
        draw.line([f_xy, m_xy], fill=color, width=LINE_W)
    for pid, color, f_xy, m_xy in pts:
        for x, y in (f_xy, m_xy):
            draw.ellipse([x - R, y - R, x + R, y + R], fill=color, outline=(255, 255, 255), width=2)
        for x, y in (f_xy, m_xy):
            draw.text((x + R + 3, y - R - 3), str(pid), fill=(0, 0, 0), font=font_label,
                      stroke_width=2, stroke_fill=(255, 255, 255))
    multi = sum(1 for j in jobs if j[0] == cam) > 1
    out_name = sanitize(cam) + ("__" + sanitize(label) if multi else "") + ".png"
    if out_name in seen:
        raise SystemExit(f"duplicate output name {out_name}")
    seen.add(out_name)
    canvas.save(os.path.join(IMG_DIR, out_name))
    manifest.append({"file": out_name, "camera": cam, "source": src_name, "source_size": [W, H],
                     "crop_floorplan": [x0, y0, x1, y1], "crop_size": [cw, ch], "canvas": [canvas_w, canvas_h],
                     "pairs_drawn": [p["id"] for p in drawn], "pairs_skipped": skipped,
                     "framePt_out_of_bounds": oob})

unique_drawn = sum(1 for pairs in pairs_by_cam.values() for p in pairs
                   if p.get("framePt") is not None and p.get("mapPt") is not None)
unique_skipped = [(c, p["id"], "framePt null") for c, pairs in pairs_by_cam.items() for p in pairs
                  if p.get("framePt") is None]
total_bytes = sum(os.path.getsize(os.path.join(IMG_DIR, m["file"])) for m in manifest)
with open(OUT + "/manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=1)
print("outputs:", len(manifest), "| unique pairs drawn:", unique_drawn, "| skipped:", unique_skipped)
print("problems:", problems)
print("total png bytes:", total_bytes, "| export sha256:", sha256(EXPORT)[:12], "| floorplan sha256:", sha256(FLOOR)[:12])
