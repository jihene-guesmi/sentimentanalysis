import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import torch

from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForSequenceClassification

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="NLP Sentiment App", layout="centered")

st.title("🎬 NLP Sentiment Analysis")
st.write("Compare LSTM vs BERT on movie reviews")

# =========================
# LOAD LSTM MODEL + TOKENIZER
# =========================
@st.cache_resource
def load_lstm():
    return tf.keras.models.load_model("lstm_model.h5")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

lstm_model = load_lstm()
tokenizer = load_tokenizer()

# =========================
# LOAD BERT MODEL
# =========================
@st.cache_resource
def load_bert():
    bert_tokenizer = BertTokenizer.from_pretrained("bert_finetuned_model")
    bert_model = BertForSequenceClassification.from_pretrained(
        "bert_finetuned_model"
    )
    bert_model.eval()
    return bert_tokenizer, bert_model

bert_tokenizer, bert_model = load_bert()

# =========================
# LSTM PREDICTION
# =========================
def predict_lstm(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=200, padding="post", truncating="post")

    score = lstm_model.predict(padded)[0][0]
    return score

# =========================
# BERT PREDICTION
# =========================
def predict_bert(text):
    inputs = bert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=200
    )

    with torch.no_grad():
        outputs = bert_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
    return probs

# =========================
# USER INPUT
# =========================
text = st.text_area("✍️ Enter a movie review:")

mode = st.selectbox("Choose model", ["LSTM", "BERT", "Compare Both"])

# =========================
# PREDICTION
# =========================
if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter a review.")
    else:

        # ---------------- LSTM ----------------
        if mode == "LSTM":
            score = predict_lstm(text)

            st.write("Raw score:", score)

            if score > 0.6:
                st.success(f"😊 Positive ({score:.2f})")
            else:
                st.error(f"😡 Negative ({score:.2f})")

        # ---------------- BERT ----------------
        elif mode == "BERT":
            probs = predict_bert(text)
            label = np.argmax(probs)

            st.write("Raw probabilities:", probs)

            if label == 1:
                st.success(f"😊 Positive ({probs[1]:.2f})")
            else:
                st.error(f"😡 Negative ({probs[0]:.2f})")

        # ---------------- COMPARE ----------------
        else:
            st.subheader("📊 Comparison Result")

            # LSTM
            lstm_score = predict_lstm(text)
            st.write("### LSTM")
            st.write("Score:", lstm_score)

            if lstm_score > 0.6:
                st.success("Positive")
            else:
                st.error("Negative")

            # BERT
            probs = predict_bert(text)
            label = np.argmax(probs)

            st.write("### BERT")
            st.write("Probabilities:", probs)

            if label == 1:
                st.success("Positive")
            else:
                st.error("Negative")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🔬 NLP Project: LSTM vs BERT Sentiment Analysis")