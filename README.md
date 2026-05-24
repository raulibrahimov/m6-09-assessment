![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# m6-09 — Cat Detection v2

Final assessment for Unit 6: improve the YOLO26 cat detector from `m6-04-assessment`, export to ONNX, ship as a Docker image with a fixed CLI.

See `m6-09-assessment.ipynb` for the training/improvement work and `container/` for the shipped inference image.

## Image for leaderboard

```text
docker pull raulito7/cat-detector:final
Image: raulito7/cat-detector:final
Student: Raul Ibrahimov
```

> Replace `raulito7` once the image is pushed.

## Local sanity check

```bash
docker build -t raulito7/cat-detector:final -f container/Dockerfile .

docker run --rm raulito7/cat-detector:final info

mkdir -p /tmp/inp /tmp/out
cp some/test/images/*.jpg /tmp/inp/
docker run --rm \
  -v /tmp/inp:/data/input:ro \
  -v /tmp/out:/data/output \
  raulito7/cat-detector:final predict

head /tmp/out/predictions.csv
```

---

# Final Assessment | Cat Detection v2 — Improve, Export to ONNX, Containerise & Compete

## Overview

This is the **final assessment** of Unit 6, and it has three parts that fit together:

1. **Improve** the YOLO26 cat detector you trained in the Week-1 assessment using techniques from Week 2 (larger backbones, stronger augmentation, transfer learning, hyperparameter tuning, smart regularisation).
2. **Convert** the improved model to **ONNX** using YOLO26's NMS-free end-to-end export, so the deployed model is production-ready and framework-agnostic.
3. **Containerise** the inference logic in a Docker image with a **fixed, standardised CLI** that the instructor will run on an unseen holdout set during the class activity. Your image will be scored on mAP@0.5 and ranked on a live leaderboard.

The dataset is the same as Week 1:

- [Cat Detection Dataset on Google Drive](https://drive.google.com/drive/folders/1qeGvkaK7UkNMYoESQHxGbV4DRH8EgEb0?usp=drive_link)

## Container CLI contract (summary)

- `docker run --rm <image> info` → prints `STUDENT.json` to stdout.
- `docker run --rm -v <input>:/data/input:ro -v <output>:/data/output <image> predict` → writes `/data/output/predictions.csv` with header `image_path,xmin,ymin,xmax,ymax,confidence,class`. One row per detection; for images with no detections, a single row with the filename and empty box fields.

See the original assessment brief for the full spec.
