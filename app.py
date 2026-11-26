import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

HF_MODEL_ID = "abhinendra9792/distilbert-github-issue-classifier"
ID2LABEL = {0: "Bug", 1: "Feature Request", 2: "Documentation"}

st.set_page_config(page_title="GitHub Issue Triage (NLP-46)", page_icon="🛠️", layout="wide")

def clean_github_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"```[\s\S]*?```", " [CODE] ", text)
    text = re.sub(r"`[^`]*`", " [CODE] ", text)
    text = re.sub(r"http\S+|www\.\S+", " [URL] ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[#*_>\[\]\(\)]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

@st.cache_resource
def load_assets():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    bert_model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
    bert_model.eval()
    baseline_model = joblib.load("tfidf_baseline.joblib")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    ref_df = pd.read_csv("reference_issues.csv")
    ref_embeddings = np.load("issue_embeddings.npy")
    return tokenizer, bert_model, baseline_model, sbert_model, ref_df, ref_embeddings

tokenizer, bert_model, baseline_model, sbert_model, ref_df, ref_embeddings = load_assets()

# Sidebar controls
st.sidebar.header("⚙️ Triage Settings")
model_choice = st.sidebar.radio(
    "Classification Model",
    ["DistilBERT (Fine-Tuned)", "TF-IDF + Logistic Regression (Baseline)"]
)
dup_threshold = st.sidebar.slider("Duplicate Similarity Threshold", 0.50, 0.95, 0.70, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Benchmark (Test Set)**")
st.sidebar.metric("DistilBERT Macro F1", "75.79%", "+6.03% vs Baseline")
st.sidebar.metric("TF-IDF Baseline Macro F1", "69.76%")

# Main Header
st.title("🛠️ GitHub Issue Classifier & Duplicate Bug Detector")
st.caption("CSB306 NLP Project (NLP-46) — Automated Issue Triage using DistilBERT & SBERT")

# Quick-fill sample buttons for live demo
col_b1, col_b2, col_b3 = st.columns(3)
if col_b1.button("🐛 Sample Bug"):
    st.session_state["title"] = "App crashes with NullPointerException on login"
    st.session_state["body"] = "When I click the login button without entering a password, the Android app immediately crashes."
if col_b2.button("✨ Sample Feature Request"):
    st.session_state["title"] = "Add dark mode toggle to settings page"
    st.session_state["body"] = "It would be great to have a dark theme option so users can switch UI themes at night."
if col_b3.button("📚 Sample Documentation"):
    st.session_state["title"] = "Update API installation guide for Python 3.12"
    st.session_state["body"] = "The README setup instructions are outdated and missing the required pip install flags."

title_input = st.text_input("Issue Title", value=st.session_state.get("title", ""), placeholder="e.g., App crashes on clicking submit")
body_input = st.text_area("Issue Description", value=st.session_state.get("body", ""), height=140, placeholder="Paste steps to reproduce, error logs, or feature details...")

if st.button("🔍 Analyze Issue", type="primary"):
    if not title_input.strip() and not body_input.strip():
        st.warning("Please enter an issue title or description.")
    else:
        cleaned = clean_github_text(f"{title_input} {body_input}")

        # 1. Classification
        if model_choice.startswith("DistilBERT"):
            inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                probs = torch.softmax(bert_model(**inputs).logits, dim=-1).numpy()[0]
        else:
            probs = baseline_model.predict_proba([cleaned])[0]

        pred_idx = int(np.argmax(probs))
        pred_label = ID2LABEL[pred_idx]

        # 2. Duplicate Detection (SBERT + Cosine Similarity)
        q_emb = sbert_model.encode([cleaned], normalize_embeddings=True)
        scores = cosine_similarity(q_emb, ref_embeddings)[0]
        top3_idx = np.argsort(scores)[::-1][:3]
        top_score = float(scores[top3_idx[0]])

        st.markdown("---")
        left, right = st.columns([1, 1.3])

        with left:
            st.subheader("🏷️ Predicted Category")
            color_map = {"Bug": "🔴", "Feature Request": "🟢", "Documentation": "🔵"}
            st.success(f"{color_map[pred_label]} **{pred_label}** ({probs[pred_idx]*100:.1f}% confidence)")
            st.write("**Class Probabilities:**")
            for idx, lbl in ID2LABEL.items():
                st.progress(float(probs[idx]), text=f"{lbl}: {probs[idx]*100:.1f}%")

        with right:
            st.subheader("🔎 Top-3 Similar Existing Issues")
            if top_score >= dup_threshold:
                st.error(f"⚠️ **Potential Duplicate Detected!** Top match similarity is **{top_score*100:.1f}%** (Threshold: {dup_threshold*100:.0f}%).")
            else:
                st.info(f"✅ No duplicate above {dup_threshold*100:.0f}% threshold. Highest match: **{top_score*100:.1f}%**.")

            for rank, idx in enumerate(top3_idx, 1):
                sim = float(scores[idx]) * 100
                row = ref_df.iloc[idx]
                with st.expander(f"#{rank} — [{sim:.1f}% Match] ({row['clean_label']}) {str(row['title'])[:85]}"):
                    st.write(str(row["body"])[:400] + "...")