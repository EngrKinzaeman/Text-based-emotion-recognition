# emotion_classification.py
# ==========================================
# EMOTION CLASSIFICATION USING MACHINE LEARNING
# (Model & Graph Saving Included)
# ==========================================

import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# ==========================================
# STEP 1: LOAD DATA
# ==========================================
df = pd.read_csv("Dataset.csv")

print("✅ Dataset loaded successfully!")
print(df.head())

# ==========================================
# STEP 2: BASIC PREPROCESSING
# ==========================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

df['clean_text'] = df['text'].apply(clean_text)

# ==========================================
# STEP 3: TRAIN-TEST SPLIT (70% / 30%)
# ==========================================
X = df['clean_text']
y = df['Emotion']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"🧩 Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

# ==========================================
# STEP 4: FEATURE EXTRACTION (TF-IDF)
# ==========================================
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ==========================================
# STEP 5: TRAIN MODEL
# ==========================================
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ==========================================
# STEP 6: EVALUATION METRICS
# ==========================================
y_pred = model.predict(X_test_tfidf)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ==========================================
# STEP 7: CONFUSION MATRIX (Display + Save)
# ==========================================
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='viridis', xticks_rotation=45)
plt.title("Emotion Classification - Confusion Matrix")
plt.tight_layout()

# Create "outputs" folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Save confusion matrix image
cm_path = os.path.join("outputs", "confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
print(f"📁 Confusion matrix saved as: {cm_path}")

plt.show()

# ==========================================
# STEP 8: SAVE MODEL & VECTORIZER
# ==========================================
model_path = os.path.join("outputs", "emotion_model.pkl")
vectorizer_path = os.path.join("outputs", "tfidf_vectorizer.pkl")

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"✅ Model saved at: {model_path}")
print(f"✅ Vectorizer saved at: {vectorizer_path}")

# ==========================================
# STEP 9: PREDICT FUNCTION (with confidence)
# ==========================================
def predict_emotion(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    probs = model.predict_proba(vectorized)[0]
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]
    emotion = model.classes_[pred_idx]

    if confidence < 0.4:
        emotion = "unclear/mixed"

    return emotion, round(confidence, 3)

# ==========================================
# STEP 10: TEST ON SAMPLE INPUT
# ==========================================
sample_texts = [
    "I am not happy with your explainations.",
    "I am so happy and grateful today!",
    "I feel angry and betrayed by my friends.",
    "I'm bored and tired of everything."
]

print("\n🧠 Sample Predictions:")
for txt in sample_texts:
    emotion, conf = predict_emotion(txt)
    print(f"Text: {txt}\n→ Emotion: {emotion} (Confidence: {conf})\n")
