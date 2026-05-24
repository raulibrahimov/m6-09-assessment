"""Export cats_v2_ft → ONNX and sanity-check against PyTorch."""
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

M6_09 = Path("/Users/raul/Desktop/m6-09-assessment")
WEEK1 = Path("/Users/raul/Desktop/m6-04-assessment")
DATA_DIR = Path("/Users/raul/Desktop/m6-04-assessmentold/data/DATA_CLEAN")

BEST_PT = M6_09 / "runs" / "cats_v2_ft" / "weights" / "best.pt"
TARGET_ONNX = M6_09 / "container" / "models" / "best.onnx"
assert BEST_PT.exists(), BEST_PT

# ---- export ----
pt_model = YOLO(str(BEST_PT))
onnx_path = pt_model.export(format="onnx", imgsz=640, opset=17, dynamic=False)
print(f"\nexported to: {onnx_path}")

TARGET_ONNX.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(onnx_path, TARGET_ONNX)
print(f"copied to:   {TARGET_ONNX}")
print(f"size:        {TARGET_ONNX.stat().st_size / 1024 / 1024:.1f} MB")

# ---- sanity check ----
sys.path.insert(0, str(M6_09 / "container" / "app"))
from detector import CatDetector  # noqa: E402

ort_det = CatDetector(str(TARGET_ONNX), imgsz=640, conf=0.25)
print(f"\nONNX output shape: {ort_det.output_shape}")

test_lines = (DATA_DIR / "test.txt").read_text().splitlines()
random.seed(0)
sample = random.sample(test_lines, 5)

print(f"\n{'image':<32} {'pt_n':>6} {'ort_n':>6} {'max_box_dpx':>12} {'max_score_d':>12}")
for rel in sample:
    abs_path = DATA_DIR / rel.lstrip("./")
    pt_res = pt_model.predict(str(abs_path), imgsz=640, conf=0.25, verbose=False)[0]
    if pt_res.boxes is not None and len(pt_res.boxes):
        pt_boxes = pt_res.boxes.xyxy.cpu().numpy()
        pt_scores = pt_res.boxes.conf.cpu().numpy()
    else:
        pt_boxes = np.empty((0, 4)); pt_scores = np.empty((0,))

    ort_dets = ort_det.predict(str(abs_path))
    if ort_dets:
        ort_boxes = np.array([[d["xmin"], d["ymin"], d["xmax"], d["ymax"]] for d in ort_dets])
        ort_scores = np.array([d["confidence"] for d in ort_dets])
    else:
        ort_boxes = np.empty((0, 4)); ort_scores = np.empty((0,))

    n = min(len(pt_boxes), len(ort_boxes))
    if n:
        pt_idx = np.argsort(-pt_scores)[:n]
        ort_idx = np.argsort(-ort_scores)[:n]
        box_d = float(np.abs(pt_boxes[pt_idx] - ort_boxes[ort_idx]).max())
        score_d = float(np.abs(pt_scores[pt_idx] - ort_scores[ort_idx]).max())
    else:
        box_d = score_d = float("nan")
    print(f"{abs_path.name:<32} {len(pt_boxes):>6} {len(ort_boxes):>6} {box_d:>12.4f} {score_d:>12.6f}")
