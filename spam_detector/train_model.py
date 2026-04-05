"""
Spam / Ham Email Classifier
Trains a Logistic Regression model with TF-IDF features
and saves it using joblib for later use in the Flask API.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.pipeline import Pipeline

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "spam_ham_dataset.csv")
MODEL_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "spam_model.pkl")
STATS_PATH = os.path.join(MODEL_DIR, "model_stats.json")

# ─── Load Data ────────────────────────────────────────────────────────────────
print("📂  Loading dataset...")
df = pd.read_csv(DATA_PATH, encoding="latin-1")

# Keep only the columns we need
df = df[["text", "label"]].dropna()
df["label"] = df["label"].str.lower().str.strip()   # 'ham' / 'spam'

print(f"    Total samples : {len(df)}")
print(f"    Spam          : {(df['label']=='spam').sum()}")
print(f"    Ham           : {(df['label']=='ham').sum()}")

# ─── Split ────────────────────────────────────────────────────────────────────
X = df["text"]
y = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n🔀  Train: {len(X_train)}  |  Test: {len(X_test)}")

# ─── Build Pipeline ───────────────────────────────────────────────────────────
print("\n🔧  Training model (TF-IDF + Logistic Regression)...")
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=10_000,
        ngram_range=(1, 2),     # unigrams + bigrams
        sublinear_tf=True,
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        C=5,
        solver="lbfgs",
        class_weight="balanced",
    )),
])

pipeline.fit(X_train, y_train)

# ─── Evaluate ─────────────────────────────────────────────────────────────────
y_pred      = pipeline.predict(X_test)
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_prob)
report    = classification_report(y_test, y_pred, target_names=["Ham", "Spam"], output_dict=True)
cm        = confusion_matrix(y_test, y_pred).tolist()

print(f"\n✅  Accuracy : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"    ROC-AUC  : {roc_auc:.4f}")
print("\n" + classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
print(f"    Confusion Matrix:\n    TN={cm[0][0]}  FP={cm[0][1]}\n    FN={cm[1][0]}  TP={cm[1][1]}")

# ─── Save Model ───────────────────────────────────────────────────────────────
joblib.dump(pipeline, MODEL_PATH)
print(f"\n💾  Model saved → {MODEL_PATH}")

# ─── Save Stats (for the UI) ──────────────────────────────────────────────────
import json
stats = {
    "accuracy":       round(accuracy * 100, 2),
    "roc_auc":        round(roc_auc * 100, 2),
    "total_samples":  len(df),
    "spam_count":     int((df["label"] == "spam").sum()),
    "ham_count":      int((df["label"] == "ham").sum()),
    "train_size":     len(X_train),
    "test_size":      len(X_test),
    "precision_spam": round(report["Spam"]["precision"] * 100, 2),
    "recall_spam":    round(report["Spam"]["recall"] * 100, 2),
    "f1_spam":        round(report["Spam"]["f1-score"] * 100, 2),
    "confusion_matrix": cm,
    "model":          "TF-IDF + Logistic Regression",
    "features":       10000,
    "ngrams":         "1-2",
}
with open(STATS_PATH, "w") as f:
    json.dump(stats, f, indent=2)

print(f"📊  Stats  saved → {STATS_PATH}")
print("\n🎉  Training complete!")
