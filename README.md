# 🎭 Multimodal Emotion Recognition
### Cross-Dataset Learning with Vision Transformer + Mamba

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-68.71%25-brightgreen)]()
[![F1 Score](https://img.shields.io/badge/Macro%20F1-0.64-brightgreen)]()

> Multimodal emotion recognition using **Vision Transformer (ViT)** for facial images and **Mamba (SSM)** for speech audio — aligned via **CLIP-style contrastive learning** across independently collected datasets.

---

## 🔍 Overview

Most multimodal emotion recognition systems require synchronized audio-visual recordings from the same subject — expensive and hard to scale. This project proposes a **label-aligned cross-dataset learning** paradigm that pairs:

- 🖼️ **FER2013** — 35,887 facial expression images (7 emotion classes)
- 🔊 **RAVDESS** — 1,440 speech recordings (same 7 emotion classes)

...matched **only by emotion label**, without identity correspondence or temporal synchronization.

```
FER2013 Image [Happy] + RAVDESS Audio [Happy] → Aligned Emotion Representation
```

---

## 🏗️ Architecture

![Architecture](https://github.com/user-attachments/assets/52bd082b-df19-46ab-b00d-08feb6597289)

### Key Design Choices

| Component | Choice | Reason |
|-----------|--------|--------|
| Visual encoder | ViT-Base/16 (frozen) | Global self-attention for facial structure |
| Audio encoder | Mamba-130M (frozen) | Linear-time O(n) sequence modeling |
| Alignment | CLIP-style contrastive loss | Bridge cross-modal distribution gap |
| Fusion | Concatenation + MLP | Efficient; strong baseline |
| Pairing | Label-aligned only | No synchronized data needed |

---

## 📊 Results

### Baseline Comparison

| Method | Accuracy | Precision | Recall | F1-score |
|--------|----------|-----------|--------|----------|
| Visual only (ViT) | 48.66% | 43.51% | 48.07% | 42.85% |
| Audio only (Mamba) | 31.50% | 31.88% | 31.50% | 30.53% |
| Late Fusion (no alignment) | 55.22% | 54.51% | 50.09% | 51.25% |
| **Ours (CLIP alignment)** | **68.71%** | **66.48%** | **66.39%** | **65.67%** |

### Per-Class Performance (Best Epoch)

| Emotion | Precision | Recall | F1-score | Support |
|---------|-----------|--------|----------|---------|
| Angry | 0.61 | 0.76 | 0.67 | 958 |
| Disgust | 0.40 | 0.45 | 0.42 | 111 |
| Fear | 0.67 | 0.47 | 0.55 | 1,024 |
| Happy | **0.86** | 0.75 | **0.80** | 1,774 |
| Neutral | 0.76 | **0.78** | **0.77** | 1,233 |
| Sad | 0.59 | 0.57 | 0.58 | 1,247 |
| Surprise | 0.59 | **0.79** | 0.68 | 831 |
| **Macro Avg** | **0.64** | **0.65** | **0.64** | 7,178 |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/<your-username>/multimodal-emotion-recognition.git
cd multimodal-emotion-recognition
pip install -r requirements.txt
```

### Requirements

```txt
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
transformers>=4.35.0
mamba-ssm>=1.1.0
librosa>=0.10.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
pillow>=9.5.0
tqdm>=4.65.0
```

### Dataset Setup

1. Download **FER2013** from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
2. Download **RAVDESS** from [Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
3. Place datasets in the `data/` directory:

```
data/
├── fer2013/
│   ├── train/
│   │   ├── angry/
│   │   ├── happy/
│   │   └── ...
│   └── test/
└── ravdess/
    ├── Actor_01/
    ├── Actor_02/
    └── ...
```

### Training

```bash
python src/train.py \
  --fer2013_path data/fer2013 \
  --ravdess_path data/ravdess \
  --epochs 15 \
  --batch_size 16 \
  --lr 3e-4
```

### Evaluation

```bash
python src/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --fer2013_path data/fer2013/test
```

### Demo (Single Prediction)

```bash
python src/demo.py \
  --image path/to/face.jpg \
  --audio path/to/speech.wav \
  --checkpoint checkpoints/best_model.pt
```

---

## 📁 Project Structure

```
multimodal-emotion-recognition/
├── src/
│   ├── models/
│   │   ├── visual_encoder.py     # ViT backbone + projection head
│   │   ├── audio_encoder.py      # Mamba backbone + projection head
│   │   ├── classifier.py         # MLP fusion classifier
│   │   └── framework.py          # Full multimodal model
│   ├── data/
│   │   ├── fer2013_dataset.py    # FER2013 dataloader
│   │   ├── ravdess_dataset.py    # RAVDESS dataloader + Mel spectrogram
│   │   └── paired_dataset.py    # Label-aligned pairing logic
│   ├── train.py                  # Training loop
│   ├── evaluate.py               # Evaluation script
│   └── demo.py                   # Single-sample inference
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_analysis.ipynb
│   └── 03_results_visualization.ipynb
├── configs/
│   └── default.yaml              # Training hyperparameters
├── assets/
│   └── architecture.png          # Framework diagram
├── requirements.txt
└── README.md
```

---

## ⚙️ Training Details

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Learning rate | 3 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁵ |
| Batch size | 16 |
| Epochs | 15 |
| Contrastive temperature (τ) | 0.07 |
| Training pairs | 2,100 (300 per class) |
| Validation set | FER2013 test (7,178 images) |

### Loss Schedule

The contrastive loss weight α(t) follows a staged schedule to stabilize training:

```
Epochs 1–5:   α = 0.08 × t   (warm-up)
Epochs 6–10:  α = 0.40       (stable)
Epochs 11–15: α = 0.25       (decay)
```

---

## 🔬 Method Highlights

### Label-Aligned Cross-Dataset Pairing

Unlike conventional MER that requires synchronized recordings, we pair samples **by emotion label only**:

- ✅ No identity matching required
- ✅ No temporal synchronization required
- ✅ Can leverage any existing unimodal datasets
- ✅ Scales with dataset size independently per modality

### Why Mamba for Audio?

Transformer-based audio encoders (WavLM, HuBERT) face **O(n²) complexity** with sequence length. Mamba's selective state space mechanism provides:

- **O(n) linear-time** processing
- 5× faster inference vs. transformers on long sequences
- Competitive performance with significantly less memory

### CLIP-Style Contrastive Alignment

We align cross-modal embeddings via bidirectional contrastive loss, pushing same-emotion pairs together and different-emotion pairs apart — even without identity-matched recordings.

---

## 📈 Limitations & Future Work

**Current limitations:**
- Small audio dataset (RAVDESS: 1,440 clips) limits audio representation diversity
- Domain gap between FER2013 (unconstrained) and RAVDESS (acted, controlled)
- Simple concatenation fusion doesn't model explicit cross-modal interactions

**Planned improvements:**
- [ ] Add CREMA-D and IEMOCAP for richer audio representations
- [ ] Experiment with cross-attention fusion mechanisms
- [ ] Partial fine-tuning of higher ViT/Mamba layers
- [ ] Real-time inference demo with webcam + microphone

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{sawatdeenarunat2025multimodal,
  title     = {Emotion Recognition from Facial and Audio Modalities},
  author    = {Sawatdeenarunat, Natthawat and Oumphuak, Romteera and 
               Tubcharoen, Chollada and Thossaroj, Nuntawat},
  year      = {2025},
  school    = {National Institute of Development Administration (NIDA)}
}
```

### References

- Dosovitskiy et al. (2021). *An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale.* ICLR.
- Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.
- Song & Cho (2025). *Leveraging CLIP Encoder for Multimodal Emotion Recognition.* WACV.
- FER2013 Dataset: https://www.kaggle.com/datasets/msambare/fer2013
- RAVDESS Dataset: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

---

## 👥 Authors

| Name | Student ID | Email |
|------|-----------|-------|
| Natthawat Sawatdeenarunat | 6710422011 | 6710422011@stu.nida.ac.th |
| Romteera Oumphuak | 6710422022 | 6710422022@stu.nida.ac.th |
| Chollada Tubcharoen | 6710422027 | 6710422027@stu.nida.ac.th |
| Nuntawat Thossaroj | 6710422031 | 6710422031@stu.nida.ac.th |

📍 **National Institute of Development Administration (NIDA)**, Bangkok, Thailand

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Made with ❤️ at NIDA | Data Analytics and Data Science Program</i>
</p>
