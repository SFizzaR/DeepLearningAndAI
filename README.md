# 🧠 Deep Learning for Perception

A collection of deep learning assignments exploring both core concepts and real-world applications. This repository includes:

- YOLOv5 pretrained model inference and YOLOv5 object detection (trained on **Roboflow Ingredients Detection**)
- Manual forward & backward propagation using basic neural networks
- Word2Vec embedding models: Skip-gram & CBOW (from scratch)

## 🧮 Forward & Backward Propagation (Manual)

Implemented a 2-layer feedforward neural network **from scratch** using NumPy — no frameworks, just pure math and matrices.

### 🔧 Features
- Sigmoid activation
- Binary cross-entropy loss
- Gradient descent updates
- Manual backpropagation (calculated derivatives)
---


## 🍅 YOLOv5 – Ingredients Detection (Custom Dataset)

Implemented object detection using **YOLOv5**, trained on the [Roboflow Ingredients Detection]([https://universe.roboflow.com/](https://universe.roboflow.com/test-model-rqcm2/ingredients-detection-yolov8)) dataset. This model can recognize ingredients like tomatoes, garlic, onions, and more in real-time.


### 📁 Files Overview

| File | Description |
|------|-------------|
| `Trainingmodel.ipynb` | Custom training on annotated dataset |
| `Yolo.ipynb` | Inference and visualization using trained model |


### 🏋️‍♂️ Training Setup

We trained the model using YOLOv5s (small variant) for speed and accuracy balance.

#### 🚀 Training Command

```bash
!python train.py --img 640 --batch 16 --epochs 50 \
                 --data data.yaml --weights yolov5s.pt \
                 --project runs/train --name ingredients_yolo

```

# 🧠 Word2Vec from Scratch (Skip-gram & CBOW)

This module implements Word2Vec using both **Skip-gram** and **CBOW** models in **PyTorch**, trained from scratch on a given text corpus. It includes preprocessing, embedding training, analogy reasoning, and t-SNE visualization of learned vectors.

---

## 📄 Contents

- `wordEmbeddings.py`: All tasks implemented in one file
- `text_corpus.txt`: Input corpus for training
- `word_embeddings.txt`: Learned embeddings from Skip-gram

---

## ✅ Features

### 🔹 Preprocessing
- Tokenization and punctuation removal
- Stopword filtering using NLTK
- Mapping words to integer indices

### 🔹 Skip-gram Model

- Learns to predict context from a target word
- Implemented using `nn.Embedding` + `nn.Linear`

```python
(model) --> Embedding --> Linear --> Vocab-size Output


#### Includes:

- `torch`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `opencv-python`
- `tqdm`
- `seaborn`

---

## 🙌 Acknowledgements

- **Ultralytics YOLOv5** for detection framework  
- **Roboflow** for the ingredients dataset  
- **PyTorch** for neural network models  
- **Stanford NLP** inspiration for Word2Vec  
