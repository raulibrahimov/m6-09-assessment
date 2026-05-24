"""
Fine-tune cats_v2 → cats_v2_ft, take 2.

Earlier attempt failed: aggressive aug + 2x weight_decay diverged the model
from epoch 1 onward (EarlyStopping triggered, best at epoch 1, regressed mAP
3.5 pts). This version inverts the strategy: turn augmentation OFF entirely
and use a tiny LR so we only refine the already-converged cats_v2 minimum.
"""

from pathlib import Path

from ultralytics import YOLO

WEEK1 = Path("/Users/raul/Desktop/m6-04-assessment")
M6_09 = Path("/Users/raul/Desktop/m6-09-assessment")

DATA_AUG_YAML = M6_09 / "data_aug.yaml"
CATS_V2_BEST = WEEK1 / "runs" / "cats_v2" / "weights" / "best.pt"
OUT_DIR = M6_09 / "runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

assert DATA_AUG_YAML.exists(), DATA_AUG_YAML
assert CATS_V2_BEST.exists(), CATS_V2_BEST

print(f"starting from: {CATS_V2_BEST}")
print(f"data:          {DATA_AUG_YAML}")
print(f"output:        {OUT_DIR / 'cats_v2_ft'}")

model = YOLO(str(CATS_V2_BEST))

model.train(
    data=str(DATA_AUG_YAML),
    epochs=30,
    patience=10,
    imgsz=640,
    batch=16,
    device="mps",
    project=str(OUT_DIR),
    name="cats_v2_ft",
    exist_ok=True,
    seed=42,
    # ultra-low LR continued training + cosine schedule.
    # NB: must set optimizer explicitly — `optimizer=auto` overrides lr0.
    optimizer="AdamW",
    lr0=0.0001,
    lrf=0.01,
    momentum=0.9,
    cos_lr=True,
    weight_decay=0.0005,
    # augmentation OFF — pure refinement
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    fliplr=0.5,   # keep — cats are L/R symmetric
    flipud=0.0,
    erasing=0.0,
    close_mosaic=0,
    auto_augment=None,
)
