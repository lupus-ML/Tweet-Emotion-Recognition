# 🧠 Tweet Emotion Recognition | Deep Learning Project

## Overview

This project implements an **end-to-end emotion recognition system** for short text messages (tweets).  
A deep learning model is trained to classify text into one of six emotional categories:

**anger, fear, joy, love, sadness, surprise**

The project covers the **complete machine learning lifecycle**, from data auditing and exploratory analysis to baseline modeling, deep learning, error analysis, and deployment through an interactive web application.

---

## Project Objectives

The objectives of this project are:

- To build a robust multi-class text classification model using deep learning
- To compare a deep learning approach against a classical baseline
- To analyze model behavior beyond accuracy (calibration, bias, imbalance)
- To deploy the trained model in an interactive, user-facing web application

---

## Dataset

The dataset consists of short, tweet-like text messages annotated with emotional labels.

The dataset is used for educational and research reasons ONLY. The dataset was provided by:

@inproceedings{saravia-etal-2018-carer,
    title = "{CARER}: Contextualized Affect Representations for Emotion Recognition",
    author = "Saravia, Elvis  and
      Liu, Hsien-Chi Toby  and
      Huang, Yen-Hao  and
      Wu, Junlin  and
      Chen, Yi-Shin",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D18-1404",
    doi = "10.18653/v1/D18-1404",
    pages = "3687--3697",
    abstract = "Emotions are expressed in nuanced ways, which varies by collective or individual experiences, knowledge, and beliefs. Therefore, to understand emotion, as conveyed through text, a robust mechanism capable of capturing and modeling different linguistic nuances and phenomena is needed. We propose a semi-supervised, graph-based algorithm to produce rich structural descriptors which serve as the building blocks for constructing contextualized affect representations from text. The pattern-based representations are further enriched with word embeddings and evaluated through several emotion recognition tasks. Our experimental results demonstrate that the proposed method outperforms state-of-the-art techniques on emotion recognition tasks.",

### Dataset Characteristics
- Informal language and short sentences
- Six emotion classes
- Strong class imbalance
- High semantic overlap between emotions

### Preprocessing Steps
- Text cleaning and normalization
- Label encoding via `label_mapping.json`
- Stratified splitting into train, validation, and test sets
- Export to CSV for reproducibility

---

## Project Structure
Tweet_Emotion_Recognition/
│
├── app/                          # Interactive web application
│   ├── server.py                 # FastAPI backend (model inference API)
│   └── static/                   # Frontend assets
│       ├── index.html            # UI entry point
│       ├── styles.css            # Styling (dark theme)
│       └── app.js                # Frontend logic (fetch + rendering)
│
├── artifacts/                    # Model outputs & reports
│   ├── models/
│   │   └── emotion_classifier_tf.keras
│   └── reports/
│       ├── classification_report.csv
│       └── metrics_summary.json
│
├── data/
│   ├── raw/
│   │   └── merged_training.pkl
│   └── processed/
│       ├── emotion_dataset.csv
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── label_mapping.json
│
├── figures/                      # Exploratory & diagnostic figures
│   ├── 1_distribution_of_emotions.png
│   ├── 2_distribution_of_words.png
│   ├── 3_text_length_by_emotion.png
│   └── 4_density_text_length_by_emotion.png
│
├── Figures - Notebook 2/
│   └── ConfusionMatrix.png
│
├── Figures - Notebook 3/
│   ├── Accuracy_Over_Epochs.png
│   ├── Loss_Over_Epochs.png
│   ├── ReliabilityDiagram.png
│   └── cm.png
│
├── notebooks/
│   ├── 01_Data_Audit_EDA.ipynb
│   ├── 02_Baseline_Modeling.ipynb
│   └── 03_DPL.ipynb
│
├── scripts/
│   └── pkl_to_csv.py
│
├── Report and Presentation/
│   └── 1_Report_Data_Audit_EDA.pdf
│
├── requirements.txt
└── README.md

---

## Modeling Approach

### Baseline Model (Notebook 02)

A classical machine learning baseline was implemented to establish a strong reference point.

- Text vectorization using bag-of-words
- Linear classification model
- Test accuracy of approximately **91%**

**Limitations**
- Strong lexical bias
- Limited contextual understanding
- Overconfidence on frequent tokens

---

### Deep Learning Model (Notebook 03)

A neural network pipeline was constructed using TensorFlow/Keras.

#### Architecture
1. TextVectorization layer (train-only adaptation)
2. Embedding layer (128 dimensions)
3. Global Average Pooling
4. Dense layer with dropout
5. Softmax output layer

#### Training Details
- Loss: `sparse_categorical_crossentropy`
- Optimizer: Adam
- Learning rate scheduling
- Validation-based monitoring

---

## Evaluation Results

### Deep Learning Model Performance
- **Test Accuracy:** ~89.4%
- Slightly lower than the baseline but more expressive and flexible

### Key Observation
Accuracy alone was found to be insufficient for understanding true model behavior.  
Extensive diagnostic analysis was therefore conducted.

---

## Error Analysis & Diagnostics

### Learning Curves
- Training loss decreases steadily
- Validation loss stabilizes and slightly increases
- Mild overfitting observed

### Confidence vs Correctness
- High-confidence predictions are mostly correct
- Medium-confidence predictions show miscalibration
- Overconfidence is observed in certain error regions

### Lexical Bias
- Frequent tokens dominate predictions across classes
- Emotion-specific words overlap heavily
- Ambiguous language causes systematic confusion

### Class Imbalance Effects
- Minority classes show lower recall
- Majority classes dominate probability mass
- Confusion matrix reveals structured misclassification patterns

---

## Deployment: Interactive Web Application

The trained model was deployed using **FastAPI** with a lightweight **HTML/CSS/JavaScript frontend**.

### Application Features
- User-entered text prediction
- Emotion label with confidence score
- Full probability distribution visualization
- Real-time inference

---

## How to Run the Application

### 1. Install Dependencies
```bash
pip install -r requirements.txt

cd Tweet_Emotion_Recognition
uvicorn app.server:app --reload

http://127.0.0.1:8000