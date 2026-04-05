"""
Export trained model weights to JSON so the model
can run entirely in the browser (no server needed).
"""

import os
import json
import joblib
import numpy as np

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "spam_detector", "spam_model.pkl")
OUT_PATH   = os.path.join(BASE_DIR, "docs", "model.json")
STATS_PATH = os.path.join(BASE_DIR, "spam_detector", "model_stats.json")

os.makedirs(os.path.join(BASE_DIR, "docs"), exist_ok=True)

print("📦  Loading model...")
pipeline = joblib.load(MODEL_PATH)

tfidf = pipeline.named_steps["tfidf"]
clf   = pipeline.named_steps["clf"]

# ── TF-IDF vocabulary (word → index) ──────────────────────
vocab = {k: int(v) for k, v in tfidf.vocabulary_.items()}  # term -> int index
idf   = tfidf.idf_.tolist()       # list of idf weights per feature

# ── Logistic Regression weights ───────────────────────────
coef      = clf.coef_[0].tolist()   # one weight per feature
intercept = float(clf.intercept_[0])

# ── Model stats ───────────────────────────────────────────
with open(STATS_PATH) as f:
    stats = json.load(f)

payload = {
    "vocab":        vocab,
    "idf":          idf,
    "coef":         coef,
    "intercept":    intercept,
    "sublinear_tf": True,
    "norm":         "l2",
    "stats":        stats,
}

with open(OUT_PATH, "w") as f:
    json.dump(payload, f, separators=(",", ":"))  # compact JSON

size_kb = os.path.getsize(OUT_PATH) / 1024
print(f"✅  Exported model → {OUT_PATH}")
print(f"    Vocabulary size : {len(vocab):,} terms")
print(f"    File size       : {size_kb:.1f} KB")
