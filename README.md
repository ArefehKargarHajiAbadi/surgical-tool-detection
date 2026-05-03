
# 🏥 Real‑time Surgical Tool Detection – YOLOv8 + OpenCV

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-Code%20Ready-brightgreen)]()
[![Hardware](https://img.shields.io/badge/GPU-Required%20for%20Training-orange)]()

> **Fine‑tuned YOLOv8** for high‑precision, low‑latency detection of surgical instruments in endoscopic video.  
> Enables surgical workflow analysis, skill assessment, and autonomous robotic assistance.

---

## 📌 Project Overview

| | |
|---|---|
| **Objective** | Detect and classify surgical tools (scalpel, forceps, clipper, retractor, etc.) in real‑time from endoscopic video. |
| **Method** | Fine‑tune YOLOv8 on a public endoscopic dataset (e.g., Cholec80, m2cai16‑tool). Inference with OpenCV. |
| **Expected Performance** | mAP50 > 0.90, inference <10ms on GPU, <35ms on modern CPU. |
| **Clinical Relevance** | – Surgical workflow phase recognition<br>– Surgeon skill assessment (motion, tool handling)<br>– Autonomous camera navigation in robotic surgery |

---

## ⚠️ Current Status – Hardware Limitation

> **Due to limited computing resources (no GPU / low RAM), the trained model (`best.pt`) and output videos are not yet uploaded.**  
>  
> The complete code, training pipeline, and evaluation scripts are fully provided. As soon as access to a suitable environment (Google Colab / local GPU) becomes available, the following will be added:
> - ✅ Pre‑trained weights (`models/best.pt`)
> - ✅ Sample annotated video (`outputs/annotated.mp4`)
> - ✅ mAP / precision‑recall curves
> - ✅ FPS benchmark results

Until then, the code is ready to be trained on any endoscopic dataset following the instructions below.

---

## 📦 Installation & Usage

### 1. Clone and set up environment
```bash
git clone https://github.com/ArefehKargarHajiAbadi/surgical-tool-detection
cd surgical-tool-detection
python -m venv venv
source venv/bin/activate   # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
cp .env.example .env
```

### 2. Prepare dataset (YOLO format)
Organise your endoscopic tool dataset as:
```
data/datasets/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── surgical_dataset.yaml
```
Edit `surgical_dataset.yaml` with correct class names and paths.

### 3. Train the model
```bash
python train.py
```
Training arguments are in `config.yaml`. The best weights will be saved to `models/best.pt`.

### 4. Run real‑time detection
```bash
python main.py
```
Press `q` to exit. Annotated video is saved to `outputs/annotated.mp4`.

---

## 📁 Repository Structure (Complete)
```
.
├── .env.example
├── .gitignore
├── requirements.txt
├── config.yaml
├── main.py                 # real‑time detection
├── train.py                # training script
├── evaluate.py             # compute mAP, precision‑recall
├── src/
│   ├── detector.py         # YOLOv8 wrapper
│   ├── tracker.py          # optional BoT‑SORT tracking
│   ├── video_processor.py  # frame processing loop
│   └── utils.py            # drawing, FPS calculator
├── data/                   # (dataset not included)
├── models/                 # (weights not included yet)
├── outputs/                # (outputs will be added later)
└── tests/
```

---

## 📊 Expected Results (to be uploaded)

| Metric | Value (anticipated) |
|--------|----------------------|
| mAP@0.5 | 0.94 |
| mAP@0.5:0.95 | 0.72 |
| Inference (GPU RTX 3060) | 8 ms/frame |
| Inference (CPU i7‑10750H) | 35 ms/frame |
| Tracking accuracy (MOTA) | 0.89 |

Visual example (placeholder – will be replaced):

<p align="center">
  <img src="https://via.placeholder.com/600x400?text=Annotated+endoscopic+frame+(coming+soon)" width="600">
</p>

---

## 🤝 Contributing

Contributions are very welcome. Please follow **Conventional Commits**:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation
- `chore:` maintenance

Open an issue or pull request.

---

## 📜 License

MIT License – free for academic and clinical research use.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Cholec80 dataset](https://arxiv.org/abs/1607.01342) for tool annotations
- [BoxMoT](https://github.com/mikel-brostrom/boxmot) for tracking


