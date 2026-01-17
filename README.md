# Email-Spam-Detection-System-
Machine Learning–based Email Spam Detection System using TF‑IDF text features and a Multinomial Naive Bayes classifier to classify emails as spam or ham, including training pipeline, evaluation metrics, and prediction script.
# Email Spam Detection System (ML)

This project implements an Email Spam Detection System using Python, TF-IDF vectorization, and a Multinomial Naive Bayes classifier.

## Features
- Preprocessing of raw email text.
- TF-IDF feature extraction.
- Training and evaluation of a spam classifier.
- Accuracy, precision, recall, F1-score, confusion matrix.
- Function to predict whether a new email is spam or ham.

## Tech Stack
- Python, Pandas
- scikit-learn (TfidfVectorizer, MultinomialNB)
- joblib for model saving

## Dataset
The project expects a CSV file `emails.csv` with:
- `label` → "spam" or "ham" (or 1/0 which gets mapped)
- `text` → email content

You can use any public spam email dataset and rename columns accordingly.

## How to Run
```bash
pip install -r requirements.txt  # or install pandas, scikit-learn, joblib
python spam_detection.py
