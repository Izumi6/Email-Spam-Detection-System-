<div align="center">

# 📧 Email Spam Detection System

### NLP-Powered Text Classification Pipeline with TF-IDF Vectorization & Multinomial Naive Bayes

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-~97%25-brightgreen?style=flat-square" alt="Accuracy" />
  <img src="https://img.shields.io/badge/Classifier-Multinomial_Naive_Bayes-blue?style=flat-square" alt="Naive Bayes" />
  <img src="https://img.shields.io/badge/Features-TF--IDF_N--grams-purple?style=flat-square" alt="TF-IDF" />
  <img src="https://img.shields.io/badge/Inference-Sub--Millisecond-orange?style=flat-square" alt="Inference" />
</p>

*A high-accuracy natural language processing pipeline designed to classify inbound emails into Spam vs. Ham (legitimate) using probabilistic token frequency modeling and sublinear TF-IDF scaling.*

</div>

---

## 🌟 Core Highlights

- ⚡ **High-Speed Inference:** Real-time probabilistic text classification executing in less than 5ms per message.
- 📊 **Sublinear TF-IDF Feature Extraction:** Mitigates the influence of recurring non-informative tokens while boosting distinctive spam keywords.
- 🧠 **Laplace-Smoothed Naive Bayes:** Prevents zero-probability cold-start errors for out-of-vocabulary words.
- 🔄 **Production Serialization:** Pre-fitted vectorizer and classifier serialized via `joblib` for rapid deployment.

---

## 🏗️ Classification Pipeline Architecture

```mermaid
graph TD
    A["Inbound Email Message"] --> B["Text Normalization<br/>Lowercasing · Noise Stripping · Punctuation Removal"]
    B --> C["Tokenization & Stopword Filtering"]
    C --> D["TF-IDF Vector Space Transformer<br/>Unigrams & Bigrams Matrix"]
    D --> E["Multinomial Naive Bayes Model<br/>Laplace Smoothing (alpha=1.0)"]
    E --> F["Class Probability Estimator"]
    F -->|P >= 0.50| G["🚨 Marked as SPAM"]
    F -->|P < 0.50| H["✅ Verified as HAM"]
```

---

## 🔬 Mathematical Foundation

Classification decisions follow the Maximum A Posteriori (MAP) probabilistic decision boundary:

$$\hat{y} = \arg\max_{c \in \{\text{Spam}, \text{Ham}\}} \left[ \log P(c) + \sum_{i=1}^{V} x_i \cdot \log P(w_i \mid c) \right]$$

Where:
- $P(c)$ represents the class prior probability.
- $P(w_i \mid c)$ is estimated with Laplace smoothing: $\frac{N_{ci} + \alpha}{N_c + \alpha \cdot |V|}$.
- $x_i$ represents the TF-IDF feature weight for token $w_i$.

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8+
- pip

### 1. Clone & Set Up
```bash
git clone https://github.com/Izumi6/Email-Spam-Detection-System-.git
cd Email-Spam-Detection-System-
pip install pandas scikit-learn joblib
```

### 2. Train & Evaluate the Classifier
```bash
python spam_detection.py
```
*Outputs accuracy scores, confusion matrix, precision/recall curves, and exports the serialized model.*

---

## 🛠️ Technology Stack

| Layer | Technology |
|:---|:---|
| **Core Language** | Python 3.8+ |
| **NLP & Modeling** | Scikit-learn (TfidfVectorizer, MultinomialNB) |
| **Data Processing** | Pandas, NumPy |
| **Model Serialization** | Joblib |

---

## 👤 Author

**Suyash Vakhariya**  
*AI Engineer & Machine Learning Researcher*  
- **Portfolio:** [suyashvakhariya.com](https://suyashvakhariya.com)  
- **LinkedIn:** [linkedin.com/in/suyashvakhariya](https://www.linkedin.com/in/suyashvakhariya)  
- **GitHub:** [@Izumi6](https://github.com/Izumi6)  

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
