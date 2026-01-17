"""
Email Spam Detection System using TF-IDF and Machine Learning

Steps:
1. Load and clean dataset
2. Split into train and test sets
3. Convert text to TF-IDF features
4. Train Multinomial Naive Bayes model
5. Evaluate performance
6. Provide function to predict new emails
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib


def load_data(csv_path="emails.csv"):
    """
    Load dataset from a CSV file.
    Expected columns:
        - 'label': spam / ham (or 1 / 0)
        - 'text': email content
    """
    df = pd.read_csv(csv_path)

    # Map numeric labels to text if needed
    if df["label"].dtype != "object":
        df["label"] = df["label"].map({0: "ham", 1: "spam"})

    # Basic cleaning
    df = df.dropna(subset=["label", "text"])
    df["text"] = df["text"].astype(str).str.strip()

    return df


def split_data(df, test_size=0.2, random_state=42):
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def vectorize_text(X_train, X_test):
    """
    Fit TF-IDF on training data and transform both train and test.
    """
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_df=0.95,
        min_df=5,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    return tfidf, X_train_tfidf, X_test_tfidf


def train_model(X_train_tfidf, y_train):
    """
    Train Multinomial Naive Bayes classifier.
    """
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    return model


def evaluate_model(model, X_test_tfidf, y_test):
    """
    Evaluate model using accuracy, precision, recall, F1, and confusion matrix.
    """
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="spam")
    rec = recall_score(y_test, y_pred, pos_label="spam")
    f1 = f1_score(y_test, y_pred, pos_label="spam")
    cm = confusion_matrix(y_test, y_pred)

    print("=== Evaluation Metrics (MultinomialNB) ===")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return acc, prec, rec, f1, cm


def save_artifacts(model, tfidf, model_path="spam_model.pkl", vec_path="tfidf_vectorizer.pkl"):
    """
    Save trained model and TF-IDF vectorizer for later use.
    """
    joblib.dump(model, model_path)
    joblib.dump(tfidf, vec_path)
    print(f"\nModel saved to {model_path}")
    print(f"Vectorizer saved to {vec_path}")


def load_artifacts(model_path="spam_model.pkl", vec_path="tfidf_vectorizer.pkl"):
    """
    Load saved model and vectorizer.
    """
    model = joblib.load(model_path)
    tfidf = joblib.load(vec_path)
    return model, tfidf


def predict_email(text, model, tfidf):
    """
    Predict whether a single email text is spam or ham.
    """
    vec = tfidf.transform([text])
    label = model.predict(vec)[0]
    return label


def main():
    # 1. Load data
    df = load_data("emails.csv")

    # 2. Split
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. TF-IDF
    tfidf, X_train_tfidf, X_test_tfidf = vectorize_text(X_train, X_test)

    # 4. Train model
    model = train_model(X_train_tfidf, y_train)

    # 5. Evaluate
    evaluate_model(model, X_test_tfidf, y_test)

    # 6. Save artifacts
    save_artifacts(model, tfidf)

    # 7. Test prediction
    sample_email = "Congratulations! You have won a lottery. Click the link to claim your prize."
    prediction = predict_email(sample_email, model, tfidf)
    print(f"\nSample email:\n{sample_email}")
    print(f"Prediction : {prediction}")


if __name__ == "__main__":
    main()
