from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from detector import CatDetector

STUDENT_JSON = Path("/app/STUDENT.json")
MODEL_PATH = Path("/app/models/best.onnx")
INPUT_DIR = Path("/data/input")
OUTPUT_DIR = Path("/data/output")
OUTPUT_CSV = OUTPUT_DIR / "predictions.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
CSV_HEADER = ["image_path", "xmin", "ymin", "xmax", "ymax", "confidence", "class"]


def cmd_info() -> int:
    with STUDENT_JSON.open() as f:
        data = json.load(f)
    json.dump(data, sys.stdout)
    sys.stdout.write("\n")
    return 0


def cmd_predict() -> int:
    if not INPUT_DIR.is_dir():
        print(f"input dir not found: {INPUT_DIR}", file=sys.stderr)
        return 2
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detector = CatDetector(MODEL_PATH)

    image_paths = sorted(
        p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )

    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for p in image_paths:
            rel = p.relative_to(INPUT_DIR).as_posix()
            try:
                dets = detector.predict(p)
            except Exception as e:
                print(f"predict failed for {rel}: {e}", file=sys.stderr)
                writer.writerow([rel, "", "", "", "", "", ""])
                continue
            if not dets:
                writer.writerow([rel, "", "", "", "", "", ""])
                continue
            for d in dets:
                writer.writerow(
                    [
                        rel,
                        f"{d['xmin']:.2f}",
                        f"{d['ymin']:.2f}",
                        f"{d['xmax']:.2f}",
                        f"{d['ymax']:.2f}",
                        f"{d['confidence']:.4f}",
                        d["class"],
                    ]
                )
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: cli.py {info|predict}", file=sys.stderr)
        return 2
    cmd = argv[1]
    if cmd == "info":
        return cmd_info()
    if cmd == "predict":
        return cmd_predict()
    print(f"unknown subcommand: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
