import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import torch
import pandas as pd
import time
import plotly.express as px

from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Cinema AI Analyzer",
    page_icon="🎬",
    layout="centered"
)

# =========================
# SESSION STATE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# CINEMA UI STYLE
# =========================
st.markdown("""
<style>

.title {
    text-align: center;
    font-size: 46px;
    font-weight: 900;
    color: #ffcc00;
    animation: fadeIn 1s ease-in-out;
}

.subtitle {
    text-align: center;
    color: #aaa;
    font-size: 16px;
    margin-bottom: 20px;
}

/* CINEMA CARD */
.card {
    background: linear-gradient(135deg, #1b1b1b, #111);
    padding: 20px;
    border-radius: 18px;
    margin-bottom: 20px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.6);
    border: 1px solid #2a2a2a;
}

/* MOVIE RESULT CARD */
.movie-positive {
    background: linear-gradient(135deg, #0f3d2e, #145a3a);
    padding: 18px;
    border-radius: 16px;
    color: #00ff99;
    font-size: 18px;
    font-weight: 800;
    text-align: center;
    animation: popIn 0.5s ease-in-out;
}

.movie-negative {
    background: linear-gradient(135deg, #3d1a1a, #5a1f1f);
    padding: 18px;
    border-radius: 16px;
    color: #ff4d4d;
    font-size: 18px;
    font-weight: 800;
    text-align: center;
    animation: popIn 0.5s ease-in-out;
}

/* CINEMA LABEL */
.movie-tag {
    display: inline-block;
    background: #222;
    padding: 6px 12px;
    border-radius: 10px;
    margin: 5px 5px 5px 0;
    color: #ffcc00;
    font-size: 12px;
}

/* ANIMATIONS */
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

@keyframes popIn {
    0% {transform: scale(0.8); opacity: 0;}
    100% {transform: scale(1); opacity: 1;}
}

.typing {
    text-align: center;
    font-size: 18px;
    color: #ccc;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER (CINEMA STYLE)
# =========================
st.markdown("<div class='title'>🎬 Cinema AI Sentiment Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analyze movie reviews like a film critic 🍿</div>", unsafe_allow_html=True)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_lstm():
    return tf.keras.models.load_model("lstm_model.h5")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_bert():
    tok = AutoTokenizer.from_pretrained("bert_finetuned_model")
    model = AutoModelForSequenceClassification.from_pretrained("bert_finetuned_model")
    model.eval()
    return tok, model

lstm_model = load_lstm()
tokenizer = load_tokenizer()
bert_tokenizer, bert_model = load_bert()

# =========================
# SIDEBAR (CINEMA CONTROLS)
# =========================
st.sidebar.title("🎥 Cinema Controls")
mode = st.sidebar.selectbox("Choose AI Critic", ["LSTM", "BERT", "Compare Both"])

st.sidebar.markdown("""
🍿 Tip:
- Try emotional reviews
- Example: "Best movie ever!" 🎬
""")

# =========================
# INPUT CARD
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)

text = st.text_area("🎬 Write your movie review:", height=140,
                     placeholder="Example: This movie was absolutely amazing and emotional!")

predict_btn = st.button("🍿 Analyze Movie Sentiment")

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# FUNCTIONS
# =========================
def predict_lstm(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=200, padding="post", truncating="post")
    return lstm_model.predict(padded)[0][0]

def predict_bert(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=200)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    return probs

def chart(probs, title):
    fig = px.bar(
        x=["😡 Negative", "😊 Positive"],
        y=probs,
        text=np.round(probs, 3),
        title=title
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# CINEMA MOOD EMOJI
# =========================
def mood(score):
    if score > 0.75:
        return "🔥 MASTERPIECE"
    elif score > 0.6:
        return "🎬 GOOD MOVIE"
    elif score > 0.4:
        return "😐 AVERAGE"
    else:
        return "💀 BAD MOVIE"

# =========================
# MAIN FLOW
# =========================
if predict_btn:

    if text.strip() == "":
        st.warning("⚠️ Please enter a movie review.")
    else:

        # 🎬 LOADING ANIMATION
        with st.spinner("🎥 Watching your review like a film critic..."):
            time.sleep(1.5)

        st.markdown("## 🍿 Cinema Verdict")

        result = {}

        # ================= LSTM =================
        if mode in ["LSTM", "Compare Both"]:
            score = predict_lstm(text)

            st.markdown(f"""
            <div class='movie-tag'>🎬 LSTM Critic</div>
            <div class='movie-tag'>{mood(score)}</div>
            """, unsafe_allow_html=True)

            if score > 0.6:
                st.markdown(f"<div class='movie-positive'>😊 POSITIVE REVIEW</div>", unsafe_allow_html=True)
                result["LSTM"] = "😊 Positive"
            else:
                st.markdown(f"<div class='movie-negative'>😡 NEGATIVE REVIEW</div>", unsafe_allow_html=True)
                result["LSTM"] = "😡 Negative"

        # ================= BERT =================
        if mode in ["BERT", "Compare Both"]:
            probs = predict_bert(text)
            label = np.argmax(probs)

            st.markdown(f"""
            <div class='movie-tag'>🤖 BERT Critic</div>
            """, unsafe_allow_html=True)

            if label == 1:
                st.markdown(f"<div class='movie-positive'>😊 POSITIVE REVIEW</div>", unsafe_allow_html=True)
                result["BERT"] = "😊 Positive"
            else:
                st.markdown(f"<div class='movie-negative'>😡 NEGATIVE REVIEW</div>", unsafe_allow_html=True)
                result["BERT"] = "😡 Negative"

        # ================= CINEMA DELAY =================
        with st.spinner("🍿 Generating movie score cards..."):
            time.sleep(1.2)

        st.markdown("## 📊 Film Critic Scoreboard")

        if mode in ["LSTM", "Compare Both"]:
            chart([1 - score, score], "🎬 LSTM Movie Score")

        if mode in ["BERT", "Compare Both"]:
            chart(probs, "🤖 BERT Movie Score")

        # SAVE HISTORY
        st.session_state.history.append({"Review": text, **result})

# =========================
# HISTORY
# =========================
st.markdown("---")
st.markdown("## 🎞️ Review History")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)
else:
    st.info("No movies reviewed yet 🎬")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🍿 Built like a Cinema AI Studio | LSTM vs BERT")