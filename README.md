# Visual Entailment using Multimodal Cross-Attention  
*Deep Learning Assignment 2 – COSC2779 (RMIT University)*  

---

## 📘 Project Overview
This project explores **Visual Entailment (VE)** — determining whether an image *entails*, *contradicts*, or is *neutral* with respect to a textual hypothesis.  
We designed a **multimodal cross-attention model** that fuses textual and visual features to perform reasoning across both modalities.

Our approach combines:
- **BERT** for textual representation
- **EfficientNetB0** for visual feature extraction
- **Cross-Attention Layer** to align the two modalities
- **Joint classification head** for entailment prediction

---

## 🧠 Motivation
Traditional models handle language and vision separately.  
However, tasks like *“A man holding an umbrella → The image shows it’s raining”* require reasoning across both.  
This project aims to bridge that gap by designing an **interpretable**, **multimodal**, and **end-to-end** framework.

---

## 🧩 Methodology

### 1. Data
We use a **SNLI-VE-style** dataset containing `(image, premise, hypothesis, label)` triples.  
The dataset is split into **train / val / test** sets.

### 2. Architecture
- **Text Encoder:** `bert-base-uncased`  
- **Image Encoder:** `EfficientNetB0` (pretrained on ImageNet)  
- **Fusion:** Cross-attention over the joint feature space  
- **Classifier:** Two fully connected layers → 3-way softmax  

<p align="center">
  <img src="docs/model_architecture.png" width="600">
</p>

### 3. Training Strategy
Two-stage training for stability and generalization:
1. **Stage 1 (Frozen Encoders):** Train fusion + classifier only  
2. **Stage 2 (Fine-tuning):** Unfreeze top EfficientNet layers + last N BERT blocks  

**Optimizer:** AdamW with weight decay  
**Learning Rate:** Scheduled warmup and decay  
**Regularization:** Dropout + Early Stopping  

---

## 📊 Results

| Metric | Validation | Test |
|:--|:--:|:--:|
| Accuracy | ~77% | ~76% |
| ROC-AUC | 0.85 | 0.84 |
| F1 (macro) | 0.81 | 0.80 |

**Interpretability:**  
- **Grad-CAM** highlights image regions driving visual reasoning.  
- **SHAP** explains key textual tokens influencing the decision.  

<p align="center">
  <img src="docs/gradcam_example.png" width="400">
  <img src="docs/shap_tokens.png" width="400">
</p>

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/sheikhmunim/deep_learning_for_entailment_prediction-.git
cd deep_learning_assign-2

# Create environment
python -m venv venv
source venv/bin/activate   # (or venv\Scripts\activate on Windows)

# Install dependencies
pip install -r requirements.txt
