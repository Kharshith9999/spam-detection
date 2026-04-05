# 🛡️ SpamShield AI — Email Spam Detector

An ML-powered email spam classifier built with **TF-IDF + Logistic Regression**, featuring a beautiful dark-themed web interface built with Flask.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-black?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Accuracy](https://img.shields.io/badge/Accuracy-98.84%25-brightgreen?style=flat-square)

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| ✅ Accuracy | **98.84%** |
| 🎯 F1 Score (Spam) | **98.04%** |
| 📈 ROC-AUC | **99.96%** |
| 🔍 Spam Recall | **100%** |
| 📧 Trained On | **5,171 emails** |

---

## 🚀 Features

- ⚡ **Instant classification** — paste any email, get results in milliseconds
- 📊 **Probability bars** — visual spam vs ham confidence scores
- 🕑 **History panel** — track your recent checks
- 💡 **Sample emails** — built-in spam & ham examples to test
- 🌙 **Dark glassmorphism UI** — animated, modern design

---

## 🗂️ Project Structure

```
spam-detection/
│
├── spam_ham_dataset.csv        ← Training dataset (5,171 emails)
│
└── spam_detector/
    ├── train_model.py          ← Train & save the ML model
    ├── app.py                  ← Flask API backend
    ├── model_stats.json        ← Model performance stats
    └── static/
        ├── index.html          ← Web UI
        ├── style.css           ← Glassmorphism dark theme
        └── app.js              ← Frontend JavaScript
```

---

## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/Kharshith9999/spam-detection.git
cd spam-detection
```

### 2. Install dependencies
```bash
pip install pandas scikit-learn flask joblib numpy
```

### 3. Train the model
```bash
python spam_detector/train_model.py
```
> This generates `spam_detector/spam_model.pkl` (the trained model file)

### 4. Start the web app
```bash
python spam_detector/app.py
```

### 5. Open in browser
```
http://127.0.0.1:5000
```

---

## 🧠 How It Works

1. **Preprocessing** — email text is cleaned and tokenized
2. **TF-IDF Vectorization** — converts text into numerical features (10,000 features, unigrams + bigrams)
3. **Logistic Regression** — classifies email as spam (1) or ham (0)
4. **Probability scores** — outputs confidence for each class
5. **Flask API** — serves predictions via `/api/predict` (POST)

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `POST` | `/api/predict` | Classify email text |
| `GET` | `/api/stats` | Model performance stats |

**Example request:**
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You have won $1,000,000!"}'
```

**Example response:**
```json
{
  "label": "spam",
  "is_spam": true,
  "spam_prob": 99.87,
  "ham_prob": 0.13,
  "confidence": 99.87
}
```

---

## 🛠️ Tech Stack

| Layer | Tech |
|-------|------|
| ML Model | scikit-learn (LogisticRegression) |
| Feature Extraction | TF-IDF Vectorizer |
| Backend | Flask (Python) |
| Frontend | HTML, CSS (Glassmorphism), Vanilla JS |
| Fonts | Google Fonts (Inter) |

---

## 📄 License

MIT License — feel free to use and modify!

---

> Built with 🧠 Machine Learning by [Kharshith9999](https://github.com/Kharshith9999)
