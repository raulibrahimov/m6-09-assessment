from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


class CatDetector:
    def __init__(
        self,
        onnx_path: str | Path,
        imgsz: int = 640,
        conf: float = 0.25,
        class_names: tuple[str, ...] = ("cat",),
    ):
        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.imgsz = imgsz
        self.conf = conf
        self.class_names = class_names
        self.input_name = self.session.get_inputs()[0].name
        self.output_shape = self.session.get_outputs()[0].shape

    def _letterbox(self, img: Image.Image, size: int):
        orig_w, orig_h = img.size
        scale = min(size / orig_w, size / orig_h)
        new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        pad_x = (size - new_w) // 2
        pad_y = (size - new_h) // 2
        canvas.paste(resized, (pad_x, pad_y))
        return canvas, scale, (pad_x, pad_y)

    def predict(self, image_path: str | Path) -> list[dict]:
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        x, scale, (pad_x, pad_y) = self._letterbox(img, self.imgsz)
        arr = (np.asarray(x, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        out = self.session.run(None, {self.input_name: arr})[0]
        # YOLO26 e2e default head: (1, 300, 6) — [x1, y1, x2, y2, score, class]
        # Tolerate either (1, N, 6) or (N, 6).
        if out.ndim == 3:
            out = out[0]

        results: list[dict] = []
        for row in out:
            x1, y1, x2, y2, score, cls = row[:6]
            if score < self.conf:
                continue
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale
            x1 = float(max(0.0, min(orig_w, x1)))
            y1 = float(max(0.0, min(orig_h, y1)))
            x2 = float(max(0.0, min(orig_w, x2)))
            y2 = float(max(0.0, min(orig_h, y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(
                {
                    "xmin": x1,
                    "ymin": y1,
                    "xmax": x2,
                    "ymax": y2,
                    "confidence": float(score),
                    "class": self.class_names[int(cls)],
                }
            )
        return results
