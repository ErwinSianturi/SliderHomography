"""Audit spatial coordinates and aisle boundaries for critical cameras."""

import json
import numpy as np

with open("homography_visual_downloads_20261023/cameras_export.json", encoding="utf-8") as f:
    dl = json.load(f)
with open("homography_visual_downloads_20261023/detect_20261024/out/detections.json", encoding="utf-8") as f:
    dets = json.load(f)

print("=== AISLE POLYGON BOUNDARIES ===")
for r in dl.get("regions", []):
    if r.get("type") == "aisle":
        lbl = r.get("label", "")
        if any(w in lbl.lower() for w in ["noodle", "ice cream", "laundry", "cereal", "coffee", "rice"]):
            ys = [p[1] for p in r["points"]]
            xs = [p[0] for p in r["points"]]
            print(f"ID {r['id']:4d} | {lbl:35s} | X: [{min(xs):4d}, {max(xs):4d}] | Y: [{min(ys):4d}, {max(ys):4d}]")

print("\n=== CRITICAL CAMERAS BASELINE DETECTIONS STATS ===")
cams = [
    "Shelf M Wow Spagheti, Energen, Sterile Milk (4mm)",
    "Shelf M Migelas (4mm)",
    "Lorong Bawah 1",
    "Shelf M Laundry Liquid"
]

for cname in cams:
    c_dets = [d for d in dets if d["camera"] == cname]
    if c_dets:
        ys = [d["map_px"][1] for d in c_dets if d.get("map_px")]
        xs = [d["map_px"][0] for d in c_dets if d.get("map_px")]
        print(f"Camera: {cname}")
        print(f"  Detections: {len(ys)} | Mean X: {np.mean(xs):.1f} | Mean Y: {np.mean(ys):.1f} | Y range: [{min(ys):.1f}, {max(ys):.1f}]")
