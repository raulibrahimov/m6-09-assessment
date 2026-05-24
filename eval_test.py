"""Evaluate cats_v1, cats_v2, cats_v2_ft on the same test split."""
from pathlib import Path
from ultralytics import YOLO

WEEK1 = Path("/Users/raul/Desktop/m6-04-assessment")
M6_09 = Path("/Users/raul/Desktop/m6-09-assessment")
DATA_AUG = M6_09 / "data_aug.yaml"

runs = [
    ("cats_v1",    WEEK1 / "runs" / "cats_v1" / "weights" / "best.pt"),
    ("cats_v2",    WEEK1 / "runs" / "cats_v2" / "weights" / "best.pt"),
    ("cats_v2_ft", M6_09 / "runs" / "cats_v2_ft" / "weights" / "best.pt"),
]

print(f"{'run':<14} {'mAP@0.5':>10} {'mAP@.5:.95':>12} {'P':>8} {'R':>8}")
results = {}
for name, ckpt in runs:
    m = YOLO(str(ckpt)).val(
        data=str(DATA_AUG), split="test", imgsz=640, batch=16,
        device="mps", project=str(M6_09 / "runs"),
        name=f"{name}_test_m609", exist_ok=True, verbose=False,
    )
    r = (float(m.box.map50), float(m.box.map), float(m.box.mp), float(m.box.mr))
    results[name] = r
    print(f"{name:<14} {r[0]:>10.4f} {r[1]:>12.4f} {r[2]:>8.4f} {r[3]:>8.4f}")

print("\n=== delta vs cats_v1 ===")
v1 = results["cats_v1"]
for name in ("cats_v2", "cats_v2_ft"):
    r = results[name]
    d = [r[i] - v1[i] for i in range(4)]
    print(f"{name:<14} {d[0]:+10.4f} {d[1]:+12.4f} {d[2]:+8.4f} {d[3]:+8.4f}")
