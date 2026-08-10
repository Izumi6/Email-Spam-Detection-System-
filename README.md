<div align="center">

# 📧 Email Spam Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

*Machine Learning–based spam classifier using TF-IDF text features and Multinomial Naive Bayes.*

</div>

---

## 📋 Overview

An end-to-end email spam detection pipeline that classifies emails as **spam** or **ham** (legitimate) using Natural Language Processing and classical machine learning. The system uses TF-IDF vectorization for feature extraction and a Multinomial Naive Bayes classifier for fast, probabilistic classification.

## 🏗️ Pipeline Architecture

```
Raw Email Text
    ↓
Text Preprocessing (lowercasing, cleaning)
    ↓
TF-IDF Vectorization (text → numerical features)
    ↓
Multinomial Naive Bayes Classifier
    ↓
Prediction: SPAM / HAM
```

## ✨ Features

- 📊 **TF-IDF Feature Extraction** — Converts raw text into meaningful numerical vectors
- 🧠 **Multinomial Naive Bayes** — Probabilistic classifier optimized for text data
- 📈 **Comprehensive Evaluation** — Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- 🔄 **Model Persistence** — Save and load trained models with `joblib`
- 🎯 **Prediction Function** — Classify new, unseen emails in real-time

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/Izumi6/Email-Spam-Detection-System-.git
cd Email-Spam-Detection-System-

# Install dependencies
pip install pandas scikit-learn joblib

# Run the spam detector
python spam_detection.py
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | ~97% |
| **Precision** | High |
| **Recall** | High |
| **F1-Score** | High |

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.8+ |
| NLP | TF-IDF Vectorization |
| ML Model | Multinomial Naive Bayes |
| Evaluation | Scikit-learn metrics |
| Serialization | Joblib |

## 👤 Author

**Suyash Vakhariya** — AI Engineer & ML Researcher
- 🌐 [suyashvakhariya.com](https://suyashvakhariya.com)
- 💼 [LinkedIn](https://www.linkedin.com/in/suyashvakhariya)
