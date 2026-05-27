# Multimodal Emotion Recognition System

A deep learning based multimodal emotion recognition framework designed to analyze emotional states using speech, text, and fused multimodal representations.

This project evaluates how different modalities contribute to emotion recognition performance using the Toronto Emotional Speech Set (TESS) dataset.

---

# Project Overview

Human emotions are expressed not only through words, but also through vocal tone, pitch, intensity, and speaking style.

This project implements and compares three separate pipelines:

1. **Speech Pipeline**
   - Extracts MFCC acoustic features from speech audio.
   - Uses a Bi-LSTM + Attention architecture for temporal emotion modeling.

2. **Text Pipeline**
   - Uses a transformer-based BERT text encoder.
   - Evaluates semantic emotion information from textual transcripts.

3. **Fusion Pipeline**
   - Combines speech and text embeddings using an attentively gated multimodal fusion layer.
   - Dynamically adjusts modality importance during inference.

---

# Pipeline Architectures

## Speech Pipeline
Input Audio  
→ MFCC + Delta + Delta-Delta Feature Extraction  
→ Bi-LSTM Network  
→ Self-Attention Layer  
→ Emotion Classification

## Text Pipeline
Text Transcript  
→ BERT Tokenizer  
→ Fine-Tuned BERT Encoder  
→ Dense Classification Layer

## Fusion Pipeline
Speech Embeddings + Text Embeddings  
→ Gated Attention Fusion Layer  
→ Final Emotion Prediction

---

# Experimental Results

| Pipeline | Architecture | Accuracy |
|----------|-------------|----------|
| Speech Pipeline | MFCC + Bi-LSTM + Attention | 99.64% |
| Text Pipeline | Fine-Tuned BERT | ~14% |
| Fusion Pipeline | Attentive Gated Fusion | 100.00% |

---

# Important Observation

The TESS dataset primarily contains emotionally neutral lexical phrases such as:

> "Say the word back"  
> "Say the word bar"

As a result, the text modality contains very limited emotional semantic information.  
Because of this, the standalone text pipeline collapses close to random-chance performance (~14% for 7 classes).

The speech modality carries the majority of the emotional signal through:
- Pitch variation
- Energy
- Vocal intensity
- Prosodic patterns
- Temporal acoustic behavior

The fusion pipeline dynamically learns to prioritize the speech modality when textual information is weak.

---

# Limitations

The speech emotion recognition system performs strongly on controlled emotional speech datasets containing relatively clean audio recordings.

However, real-world performance may decrease in noisy environments due to:
- Background noise
- Echo and reverberation
- Low-quality microphones
- Multiple speakers
- Environmental interference

The project was primarily trained and evaluated on clean studio-quality recordings and was not specifically optimized for noise-robust deployment.

Future improvements may include:
- Noise augmentation training
- Larger real-world datasets
- Cross-dataset generalization
- Improved multimodal fusion strategies
- Real-time deployment optimization

---

# Repository Structure

```text
Multimodal-Emotion-Recognition/
│
├── models/
│   ├── speech_pipeline/
│   ├── text_pipeline/
│   └── fusion_pipeline/
│
├── data/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Dataset

This project uses the:

## Toronto Emotional Speech Set (TESS)

The dataset contains emotional speech recordings across multiple emotion classes.

Dataset download:
https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

---

# Installation

Clone the repository:

```bash
git clone https://github.com/kishanjayaprakash/Multimodal-Emotion-Recognition.git

cd Multimodal-Emotion-Recognition
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset Setup

Download the TESS dataset and place it inside:

```text
data/TESS/
```

Expected structure:

```text
data/
└── TESS/
    ├── OAF_angry/
    ├── OAF_happy/
    ├── YAF_sad/
    └── ...
```

---

# Model Weights

Pretrained `.pth` model files are not included in the repository due to GitHub storage limitations.

Place downloaded checkpoints inside:

```text
models/
models/text_pipeline/
models/fusion_pipeline/
```

---

# Running the Speech Pipeline

```bash
cd models/speech_pipeline

python train.py
```

Testing:

```bash
python test_speech.py
```

---

# Running the Text Pipeline

```bash
cd models/text_pipeline

python train.py
```

Testing:

```bash
python test_text.py
```

---

# Running the Fusion Pipeline

```bash
cd models/fusion_pipeline

python train.py
```

Testing:

```bash
python test_fusion.py
```

---

# Live Inference

Speech inference:

```bash
cd models/speech_pipeline

python infer.py
```

Fusion inference:

```bash
cd models/fusion_pipeline

python infer_fusion.py
```

---

# Evaluation Outputs

The evaluation scripts generate:
- Accuracy reports
- Confusion matrices
- Per-class accuracy tables
- Training curves
- t-SNE feature visualizations

---

# Technologies Used

- Python
- PyTorch
- HuggingFace Transformers
- Librosa
- NumPy
- Scikit-learn
- Matplotlib

---

# Key Learning Outcomes

This project demonstrates:
- Speech-based emotion recognition
- Transformer-based text analysis
- Temporal sequence modeling using Bi-LSTMs
- Attention mechanisms
- Multimodal deep learning fusion
- Cross-modal gating strategies
- Feature visualization and evaluation

---

# Author

Kishan Jayaprakash

GitHub Repository:
https://github.com/kishanjayaprakash/Multimodal-Emotion-Recognition
Google drive link of Project : https://drive.google.com/drive/folders/1Q4eUv0S8ieQp7XKGsgnzRVI9mP-3BN7C?usp=sharing
Download Binary files of the Model : https://drive.google.com/drive/folders/1bD4fco1VvfyAxxHs54QyVpb5hpi4Shbt?usp=sharing

