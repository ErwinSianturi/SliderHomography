"""Config and Data Loader Module.

Loads cameras_export.json, frame images, raw detections, baseline projections,
and store layout regions. Provides camera anomaly taxonomy based on audit records.
"""

import collections
import glob
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(WORKSPACE_DIR, "homography_visual_downloads_20261023")
MASTER_EXPORT_PATH = os.path.join(DATA_DIR, "cameras_export.json")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")
DET_DIR = os.path.join(DATA_DIR, "detect_20261024", "det")
BASELINE_DETECTIONS_PATH = os.path.join(DATA_DIR, "detect_20261024", "out", "detections.json")
FRAMES_DIR = os.path.join(DATA_DIR, "frames_src")
OVERLAY_DIR = os.path.join(DATA_DIR, "detect_20261024", "out", "images")
PAIRS_DIR = os.path.join(DATA_DIR, "images")


def sanitize_filename(s: str) -> str:
    """Sanitizes camera name into valid filename consistent with stage2_project_draw."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")

VALID_CAMERA_STATUSES = ["Unchecked", "Flagged", "Checked"]
DEFAULT_CAMERA_STATUS = "Unchecked"
CAMERA_STATUSES_FILE = "camera_statuses.json"


def normalize_camera_status(val: Any) -> str:
    """Normalizes any input/legacy status into strictly one of: 'Checked', 'Flagged', 'Unchecked'."""
    if val is None:
        return DEFAULT_CAMERA_STATUS
    s = str(val).strip().lower()
    if s in ("checked", "verified", "selesai", "ok", "valid"):
        return "Checked"
    elif s in ("flagged", "kritis", "critical", "strong", "needs verification", "anomali", "warning", "confirmed"):
        return "Flagged"
    elif s in ("unchecked", "planned", "pending", "belum", "normal"):
        return "Unchecked"
    return DEFAULT_CAMERA_STATUS


def load_camera_statuses(workspace_dir: str) -> Dict[str, str]:
    """Loads persistent camera review statuses from camera_statuses.json."""
    p = os.path.join(workspace_dir, CAMERA_STATUSES_FILE)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {k: normalize_camera_status(v) for k, v in data.items()}
        except Exception:
            pass
    return {}


def save_camera_statuses(workspace_dir: str, statuses: Dict[str, str]) -> str:
    """Persists camera review statuses to camera_statuses.json."""
    p = os.path.join(workspace_dir, CAMERA_STATUSES_FILE)
    clean_statuses = {k: normalize_camera_status(v) for k, v in statuses.items()}
    os.makedirs(workspace_dir, exist_ok=True)
    tmp_p = p + ".tmp"
    with open(tmp_p, "w", encoding="utf-8") as f:
        json.dump(clean_statuses, f, indent=2, ensure_ascii=False)
    
    # Robust atomic replace on Windows with retry
    for attempt in range(5):
        try:
            os.replace(tmp_p, p)
            break
        except PermissionError:
            if attempt == 4:
                try:
                    with open(p, "w", encoding="utf-8") as f:
                        json.dump(clean_statuses, f, indent=2, ensure_ascii=False)
                    if os.path.exists(tmp_p):
                        os.remove(tmp_p)
                except Exception:
                    pass
            else:
                import time
                time.sleep(0.05)
    return p


# Informational reference metadata without automatic severity classifications
CAMERA_REFERENCE_METADATA = {
    "Shelf M Wow Spagheti, Energen, Sterile Milk (4mm)": {
        "intended_aisle": "Instant & Import Noodles",
        "recorded_aisle": "Cereal, Oats / Coffee & Tea",
        "description": "Baseline detect_20261024 berada di Y~1730 (border Cereal/Noodles). Data historis Parquet sebelumnya terlempar ke Y~1900 (+180 px di Coffee & Tea).",
        "recommended_dy": -75.0,
        "recommended_dx": 0.0,
    },
    "Shelf M Migelas (4mm)": {
        "intended_aisle": "Instant & Import Noodles",
        "recorded_aisle": "Cereal, Oats, Granola",
        "description": "Baseline detect_20261024 berada di Y~1748 (dalam Cereal). Koreksi ~ -75 px mengembalikan deteksi ke Instant Noodles.",
        "recommended_dy": -75.0,
        "recommended_dx": 0.0,
    },
    "Lorong Bawah 1": {
        "intended_aisle": "Ice Cream",
        "recorded_aisle": "Bread & Rice / Specialty Rice",
        "description": "Baseline detect_20261024 berada di X~922, Y~2299. Area Ice Cream berada di X:[744, 821] dan Y:[2366, 2517].",
        "recommended_dy": 80.0,
        "recommended_dx": -120.0,
    },
    "Shelf M Laundry Liquid": {
        "intended_aisle": "Laundry Detergent",
        "recorded_aisle": "Household Cleaner",
        "description": "Baseline detect_20261024 berada di Y~1330. Area Laundry Detergent berada di Y:[1247, 1298]. Koreksi ~ -58 px mengembalikan deteksi ke Laundry.",
        "recommended_dy": -58.0,
        "recommended_dx": 0.0,
    },
    "Shelf M Super Bubur": {
        "intended_aisle": "Pasta & Canned Food",
        "recorded_aisle": "Pet Food & Baby Diapers",
        "description": "Proyeksi bimodal / cross-boundary leakage melintasi walkway tengah.",
        "recommended_dy": 0.0,
        "recommended_dx": 0.0,
    },
    "Shelf M Beng2 Drink (4mm)": {
        "intended_aisle": "Ready To Drink Beverages",
        "recorded_aisle": "Dairy / OUTSIDE_AISLE",
        "description": "Diskrepansi posisi fisik kamera dan koordinat tumpah ke perimeter dinding barat.",
        "recommended_dy": 0.0,
        "recommended_dx": 0.0,
    },
    "D62": {
        "intended_aisle": "Walkway Tengah / Persimpangan",
        "recorded_aisle": "Pet Food & Supplies & Baby Diapers",
        "description": "Pergeseran proyeksi sumbu X masif ke arah Timur (+350 px / ~11,35 m).",
        "recommended_dx": -350.0,
        "recommended_dy": 0.0,
    },
    "Shelf M Cold Bev (4mm)": {
        "intended_aisle": "Cold Beverage & Cut Fruits",
        "recorded_aisle": "Tersebar > 7 lorong",
        "description": "Dispersi spasial ekstrem / unclamped projection outliers (> 15 m dari kamera).",
    },
    "D41": {
        "intended_aisle": "Fresh Area Barat Laut",
        "recorded_aisle": "Vegetable 3, Fresh Fruit, Chilled Food",
        "description": "Rentang koordinat menyebar ke seluruh toko.",
    },
}

# Alias for backward-compatibility with older tests
CAMERA_ANOMALY_REGISTRY = CAMERA_REFERENCE_METADATA


class DataLoader:
    def __init__(self, master_export_path: str = MASTER_EXPORT_PATH):
        self.master_path = master_export_path
        self.export_data: Dict[str, Any] = {}
        self.baseline_detections: List[Dict[str, Any]] = []
        self.frame_to_cam: Dict[str, str] = {}
        self.cam_to_frame: Dict[str, str] = {}
        self.homography_by_cam: Dict[str, Dict[str, Any]] = {}
        self.cameras_by_id: Dict[str, Dict[str, Any]] = {}
        self.regions: List[Dict[str, Any]] = []
        self.scale_px_per_m: float = 31.4465408805
        self.floorplan_path: Optional[str] = None
        self.floorplan_size: Tuple[int, int] = (3750, 3368)

        self._load_master_export()
        self._load_baseline_detections()
        self._build_mappings()
        self._locate_floorplan()

    def _load_master_export(self):
        if not os.path.exists(self.master_path):
            raise FileNotFoundError(f"Master config not found: {self.master_path}")
        with open(self.master_path, "r", encoding="utf-8") as f:
            self.export_data = json.load(f)

        self.scale_px_per_m = float(self.export_data.get("scale", {}).get("px_per_m", 31.4465408805))
        self.homography_by_cam = {h["cameraId"]: h for h in self.export_data.get("homography", [])}
        self.cameras_by_id = {c["id"]: c for c in self.export_data.get("cameras", [])}
        self.regions = self.export_data.get("regions", [])

    def _load_baseline_detections(self):
        if os.path.exists(BASELINE_DETECTIONS_PATH):
            with open(BASELINE_DETECTIONS_PATH, "r", encoding="utf-8") as f:
                self.baseline_detections = json.load(f)

    def _build_mappings(self):
        # Build frame_map consistent with stage2_project_draw.py
        refs = collections.defaultdict(set)
        for key in ("frameZones", "frameRegions", "frameCrops", "frameCountingLines", "frameFlowArrows"):
            for it in self.export_data.get(key) or []:
                if it.get("frameImage") and it.get("cameraId"):
                    refs[it["cameraId"]].add(it["frameImage"].split("/")[-1])

        frame_cam = collections.defaultdict(set)
        for c in self.homography_by_cam:
            for n in refs.get(c, ()):
                if n.endswith("_1.jpg"):
                    frame_cam[n].add(c)

        frames = sorted(os.path.basename(p) for p in glob.glob(os.path.join(FRAMES_DIR, "*_1.jpg")))
        for base in frames:
            prefix = base[:-len("_1.jpg")]
            if base not in frame_cam and prefix in self.homography_by_cam:
                frame_cam[base].add(prefix)

        for base in frames:
            cs = frame_cam.get(base, set())
            if len(cs) == 1:
                cam = next(iter(cs))
                self.frame_to_cam[base] = cam
                self.cam_to_frame[cam] = base

        # Fallback / verification from baseline detections
        for d in self.baseline_detections:
            cam = d.get("camera")
            frame = d.get("frame")
            if cam and frame:
                self.cam_to_frame[cam] = frame
                self.frame_to_cam[frame] = cam

    def _locate_floorplan(self):
        candidates = [
            os.path.join(WORKSPACE_DIR, "floorplan.png"),
            os.path.join(DATA_DIR, "floorplan.png"),
            os.path.join(WORKSPACE_DIR, "configs", "geometry", "floorplan.png"),
            os.path.join(DATA_DIR, "images", "floorplan.png"),
        ]
        for path in candidates:
            if os.path.exists(path):
                self.floorplan_path = path
                break

    def get_camera_list(self, custom_statuses: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        result = []
        for cam_id in sorted(self.homography_by_cam.keys()):
            cam_info = self.cameras_by_id.get(cam_id, {})
            label = cam_info.get("label", cam_id)
            frame = self.cam_to_frame.get(cam_id, "")
            pairs = self.homography_by_cam[cam_id].get("pairs", [])
            valid_pairs = [p for p in pairs if p.get("framePt") is not None and p.get("mapPt") is not None]
            c_dets = [d for d in self.baseline_detections if d.get("camera") == cam_id]

            # Strictly resolve status to one of ['Checked', 'Flagged', 'Unchecked']
            # Review status is purely user-driven and must NEVER default to Checked from legacy data.
            raw_status = None
            if custom_statuses and cam_id in custom_statuses:
                raw_status = custom_statuses[cam_id]
            else:
                raw_status = DEFAULT_CAMERA_STATUS

            status = normalize_camera_status(raw_status)
            ref_info = CAMERA_REFERENCE_METADATA.get(cam_id, None)

            result.append({
                "id": cam_id,
                "label": label,
                "frame": frame,
                "pairs_count": len(valid_pairs),
                "detections_count": len(c_dets),
                "status": status,
                "review_status": status,
                "anomaly_status": status,
                "reference_info": ref_info,
                "anomaly_details": ref_info,
                "camera_meta": cam_info,
            })
        return result

    def get_raw_detections(self, frame_name: str) -> List[Dict[str, Any]]:
        base_name = frame_name[:-4] if frame_name.endswith(".jpg") else frame_name
        json_path = os.path.join(DET_DIR, f"{base_name}.json")
        if not os.path.exists(json_path):
            return []
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("detections", [])

    def get_baseline_detections_for_camera(self, camera_id: str) -> List[Dict[str, Any]]:
        return [d for d in self.baseline_detections if d.get("camera") == camera_id]

    def get_frame_path(self, frame_name: str) -> Optional[str]:
        p = os.path.join(FRAMES_DIR, frame_name)
        return p if os.path.exists(p) else None

    def get_overlay_image_path(self, camera_id: str) -> Optional[str]:
        """Returns the pre-rendered overlay image path for this camera if it exists."""
        fn = sanitize_filename(camera_id) + ".png"
        p = os.path.join(OVERLAY_DIR, fn)
        return p if os.path.exists(p) else None

    def get_pairs_image_path(self, camera_id: str) -> Optional[str]:
        """Returns the correspondence pairs side-by-side image path for this camera if it exists."""
        fn = sanitize_filename(camera_id) + ".png"
        p = os.path.join(PAIRS_DIR, fn)
        if os.path.exists(p):
            return p
        if os.path.exists(PAIRS_DIR):
            prefix = sanitize_filename(camera_id)
            for f in os.listdir(PAIRS_DIR):
                if f.startswith(prefix) and f.endswith(".png"):
                    return os.path.join(PAIRS_DIR, f)
        return None

    def get_frame_size(self, frame_name: str) -> Tuple[int, int]:
        """Returns (width, height) of the camera frame."""
        p = self.get_frame_path(frame_name)
        if p and os.path.exists(p):
            try:
                with Image.open(p) as img:
                    return img.size
            except Exception:
                pass
        return (640, 360)

    def get_crop_bounds(self, camera_id: str, margin: int = 100) -> Optional[Tuple[int, int, int, int]]:
        """Computes floorplan crop (x0, y0, x1, y1) matching stage2_project_draw geometry."""
        cdets = self.get_baseline_detections_for_camera(camera_id)
        mapped = [d["map_px"] for d in cdets if d.get("map_px")]
        if not mapped:
            return None
        xs = [p[0] for p in mapped]
        ys = [p[1] for p in mapped]
        fw, fh = self.floorplan_size
        x0 = max(0, math.floor(min(xs) - margin))
        y0 = max(0, math.floor(min(ys) - margin))
        x1 = min(int(fw), math.ceil(max(xs) + margin))
        y1 = min(int(fh), math.ceil(max(ys) + margin))
        return (x0, y0, x1, y1)
