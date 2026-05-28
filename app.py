import streamlit as st
import pandas as pd
import nltk
import random
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Optional imports (XGBoost, DL, Transformers)
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalAveragePooling1D
    dl_available = True
except ImportError:
    dl_available = False

try:
    from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
    transformers_available = True
except ImportError:
    transformers_available = False

# Download nltk resources
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
STOPWORDS = set(stopwords.words("english"))

# ------------------------
# Text Preprocessing
# ------------------------
def preprocess_text(text, lowercase=True, remove_punct=True, remove_stop=True, remove_num=True, remove_ws=True):
    if pd.isna(text):
        return ""

    if lowercase:
        text = text.lower()
    if remove_punct:
        text = re.sub(r"[^\w\s]", "", text)
    if remove_num:
        text = re.sub(r"\d+", "", text)
    if remove_ws:
        text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    if remove_stop:
        tokens = [w for w in tokens if w not in STOPWORDS]

    return " ".join(tokens)

# ------------------------
# Dataset Check
# ------------------------
def suggest_preprocessing(df, text_col):
    suggestions = []
    sample = " ".join(df[text_col].astype(str).head(20))
    if any(c.isupper() for c in sample):
        suggestions.append("Lowercasing")
    if re.search(r"[^\w\s]", sample):
        suggestions.append("Remove Punctuation")
    if re.search(r"\d", sample):
        suggestions.append("Remove Numbers")
    if "  " in sample:
        suggestions.append("Remove Extra Whitespace")
    suggestions.append("Remove Stopwords (recommended)")
    return suggestions

# ------------------------
# Data Augmentation
# ------------------------
def synonym_replacement(text):
    words = text.split()
    if not words:
        return text
    idx = random.randint(0, len(words) - 1)
    words[idx] = words[idx][::-1]  # dummy synonym replacement
    return " ".join(words)

def random_insertion(text):
    words = text.split()
    words.insert(random.randint(0, len(words)), "extra")
    return " ".join(words)

def random_swap(text):
    words = text.split()
    if len(words) > 1:
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)

def random_deletion(text, p=0.2):
    words = text.split()
    return " ".join([w for w in words if random.random() > p]) or text

def sentence_shuffling(text):
    words = text.split()
    random.shuffle(words)
    return " ".join(words)

def sentence_cropping(text, max_len=5):
    words = text.split()
    return " ".join(words[:max_len])

def noise_injection(text, noise_level=0.1):
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < noise_level:
            chars[i] = random.choice("abcdefghijklmnopqrstuvwxyz")
    return "".join(chars)

def back_translation(text):
    return text + " [BT]"  # placeholder

def paraphrasing(text):
    return text + " [Paraphrased]"  # placeholder

AUG_METHODS = {
    "Synonym Replacement": synonym_replacement,
    "Random Insertion": random_insertion,
    "Random Swap": random_swap,
    "Random Deletion": random_deletion,
    "Sentence Shuffling": sentence_shuffling,
    "Sentence Cropping": sentence_cropping,
    "Noise Injection": noise_injection,
    "Back Translation (Placeholder)": back_translation,
    "Paraphrasing (Placeholder)": paraphrasing,
}

# ------------------------
# Streamlit App
# ------------------------
st.title("📝 NLP Preprocessing, Augmentation & Modeling App")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.loc[:, ~df.columns.duplicated()]  # remove duplicate cols

    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    text_col = st.selectbox("Select Text Column", df.columns)
    label_col = st.selectbox("Select Label Column", df.columns)

    # Suggestions
    st.subheader("🔍 Suggested Preprocessing")
    suggestions = suggest_preprocessing(df, text_col)
    st.write("Suggestions:", ", ".join(suggestions))

    # Preprocessing Options
    st.subheader("⚙️ Preprocessing Options")
    lowercase = st.checkbox("Lowercasing", "Lowercasing" in suggestions)
    remove_punct = st.checkbox("Remove Punctuation", "Remove Punctuation" in suggestions)
    remove_stop = st.checkbox("Remove Stopwords", True)
    remove_num = st.checkbox("Remove Numbers", "Remove Numbers" in suggestions)
    remove_ws = st.checkbox("Remove Extra Whitespace", "Remove Extra Whitespace" in suggestions)

    if st.button("Run Preprocessing"):
        df["processed_text"] = df[text_col].astype(str).apply(
            lambda x: preprocess_text(x, lowercase, remove_punct, remove_stop, remove_num, remove_ws)
        )
        st.success("✅ Preprocessing Done!")
        st.write(df[[text_col, "processed_text"]].head())

    # Augmentation
    st.subheader("✨ Data Augmentation")
    selected_augs = st.multiselect(
        "Select Augmentation Techniques",
        list(AUG_METHODS.keys()),
        key="augmentation_multiselect"  # unique key added
    )

    if st.button("Apply Augmentation", key="apply_aug_btn"):
        # Ensure preprocessing is always applied
        if "processed_text" not in df.columns:
            df["processed_text"] = df[text_col].astype(str).apply(
                lambda x: preprocess_text(x, lowercase, remove_punct, remove_stop, remove_num, remove_ws)
            )

        # Fallback: if processed_text empty, use original
        df["processed_text"] = df["processed_text"].apply(
            lambda t: t if str(t).strip() != "" else str(t)
        )

        for aug in selected_augs:
            df[f"aug_{aug}"] = df["processed_text"].apply(AUG_METHODS[aug])

        st.success("✅ Augmentation Done!")
        st.write(df.head())

        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Augmented Dataset",
            data=csv,
            file_name="augmented_dataset.csv",
            key="download_aug_csv"  # unique key
        )



    # Modeling
    st.subheader("🤖 Train Model")
    model_choice = st.selectbox("Select Model", [
        "Logistic Regression", "Naive Bayes", "SVM", "Random Forest", "KNN"
    ] + (["XGBoost"] if xgb_available else []) +
        (["LSTM (Keras)"] if dl_available else []) +
        (["DistilBERT (Transformers)"] if transformers_available else []))

    if st.button("Train Model"):
        X = df["processed_text"] if "processed_text" in df else df[text_col].astype(str)
        y = df[label_col]

        mask = X.str.strip() != ""
        X, y = X[mask], y[mask]

        if X.empty:
            st.error("All texts are empty after preprocessing.")
            st.stop()

        # Classical ML
        if model_choice in ["Logistic Regression", "Naive Bayes", "SVM", "Random Forest", "KNN", "XGBoost"]:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            vectorizer = TfidfVectorizer()
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=200)
            elif model_choice == "Naive Bayes":
                model = MultinomialNB()
            elif model_choice == "SVM":
                model = LinearSVC()
            elif model_choice == "Random Forest":
                model = RandomForestClassifier()
            elif model_choice == "KNN":
                model = KNeighborsClassifier()
            elif model_choice == "XGBoost" and xgb_available:
                model = XGBClassifier(eval_metric="mlogloss")

            model.fit(X_train_vec, y_train)
            preds = model.predict(X_test_vec)

            st.write("📈 Accuracy:", accuracy_score(y_test, preds))
            st.text("Classification Report:\n" + classification_report(y_test, preds))

        # Deep Learning (LSTM)
        elif model_choice == "LSTM (Keras)" and dl_available:
            from tensorflow.keras.preprocessing.text import Tokenizer
            from tensorflow.keras.preprocessing.sequence import pad_sequences

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            tokenizer = Tokenizer(num_words=5000)
            tokenizer.fit_on_texts(X_train)
            X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=100)
            X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=100)

            model = Sequential([
                Embedding(input_dim=5000, output_dim=64, input_length=100),
                LSTM(64),
                Dense(1, activation="sigmoid")
            ])
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
            model.fit(X_train_seq, y_train, epochs=1, batch_size=32, verbose=1)

            loss, acc = model.evaluate(X_test_seq, y_test, verbose=0)
            st.write("📈 Accuracy:", acc)

        # Transformers (DistilBERT)
        elif model_choice == "DistilBERT (Transformers)" and transformers_available:
            st.info("⚠️ This requires a GPU for training. Using small demo only.")

            tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
            model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

            st.write("Model Loaded: DistilBERT")
            st.warning("Training skipped in demo mode (too heavy for Streamlit).")
