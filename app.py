import re
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

HF_MODEL_ID = "abhinendra9792/distilbert-github-issue-classifier"
ID2LABEL = {0: "Bug", 1: "Feature Request", 2: "Documentation"}
BADGE_STYLE = {
    "Bug": ("#ff4b4b", "rgba(255, 75, 75, 0.15)", "🐛 Bug"),
    "Feature Request": ("#2ea043", "rgba(46, 160, 67, 0.15)", "✨ Feature Request"),
    "Documentation": ("#388bfd", "rgba(56, 139, 253, 0.15)", "📚 Documentation"),
}

st.set_page_config(page_title="GitHub Issue Triage (NLP-46)", page_icon="🛠️", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .hero-card {
        background: linear-gradient(135deg, #161b22 0%, #1f2937 55%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #f0f6fc;
        margin: 0;
    }
    .hero-sub {
        color: #8b949e;
        font-size: 0.98rem;
        margin-top: 0.35rem;
        margin-bottom: 0;
    }
    .kpi-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: left;
    }
    .kpi-label { color: #8b949e; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { color: #f0f6fc; font-size: 1.55rem; font-weight: 700; margin-top: 0.15rem; }
    .kpi-delta { color: #3fb950; font-size: 0.82rem; font-weight: 600; }
    .pill-badge {
        display: inline-block;
        padding: 0.4rem 0.95rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 0.8rem;
    }
    .result-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.25rem;
    }
</style>
""", unsafe_allow_html=True)

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

def predict_issue(cleaned_text: str, model_choice: str, tokenizer, bert_model, baseline_model):
    if model_choice.startswith("DistilBERT"):
        inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            probs = torch.softmax(bert_model(**inputs).logits, dim=-1).numpy()[0]
    else:
        probs = baseline_model.predict_proba([cleaned_text])[0]
    pred_idx = int(np.argmax(probs))
    return ID2LABEL[pred_idx], float(probs[pred_idx]), probs

@st.cache_data(ttl=300)
def fetch_live_github_issues(repo_name: str, count: int = 10):
    url = f"https://api.github.com/repos/{repo_name.strip()}/issues"
    params = {"state": "open", "per_page": min(count * 2, 100)}
    headers = {"Accept": "application/vnd.github+json"}
    resp = requests.get(url, params=params, headers=headers, timeout=15)
    if resp.status_code != 200:
        return None, f"GitHub API Error ({resp.status_code}): {resp.json().get('message', 'Unable to fetch')}"
    issues = [item for item in resp.json() if "pull_request" not in item][:count]
    return issues, None

tokenizer, bert_model, baseline_model, sbert_model, ref_df, ref_embeddings = load_assets()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### ⚙️ Triage Controls")
    model_choice = st.radio(
        "Classification Engine",
        ["DistilBERT (Fine-Tuned)", "TF-IDF + Logistic Regression (Baseline)"]
    )
    dup_threshold = st.slider("Duplicate Similarity Threshold", 0.50, 0.95, 0.70, 0.05)
    st.markdown("---")
    st.markdown("**Tech Stack**")
    st.caption("• Classifier: Fine-tuned `distilbert-base-uncased`\n• Baseline: TF-IDF + Logistic Regression\n• Duplicates: `all-MiniLM-L6-v2` (SBERT) + Cosine Similarity\n• Live Data: GitHub REST API")

# ---------------- HERO BANNER ----------------
st.markdown("""
<div class="hero-card">
    <p class="hero-title">🛠️ GitHub Issue Classifier & Duplicate Bug Detector</p>
    <p class="hero-sub">CSB306 NLP Project (NLP-46) · Automated Real-Time Issue Triage powered by DistilBERT & Sentence-BERT</p>
</div>
""", unsafe_allow_html=True)

# ---------------- KPI STRIP ----------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown('<div class="kpi-card"><div class="kpi-label">DistilBERT Macro F1</div><div class="kpi-value">75.79%</div><div class="kpi-delta">↑ +6.03% vs Baseline</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown('<div class="kpi-card"><div class="kpi-label">Test Accuracy</div><div class="kpi-value">75.85%</div><div class="kpi-delta">Balanced 3-Class Split</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown('<div class="kpi-card"><div class="kpi-label">TF-IDF Baseline F1</div><div class="kpi-value">69.76%</div><div class="kpi-delta">Classical Benchmark</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">SBERT Reference Index</div><div class="kpi-value">{len(ref_df):,}</div><div class="kpi-delta">384-dim Embeddings</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["✍️ Manual Issue Triage", "🔴 Live GitHub Repo Triage"])

with tab1:
    st.caption("Quick-fill a sample issue or type your own:")
    b1, b2, b3 = st.columns(3)
    if b1.button("🐛 Sample Bug", use_container_width=True):
        st.session_state["title"] = "App crashes with NullPointerException on login"
        st.session_state["body"] = "When I click the login button without entering a password, the Android app immediately crashes."
    if b2.button("✨ Sample Feature Request", use_container_width=True):
        st.session_state["title"] = "Add dark mode toggle to settings page"
        st.session_state["body"] = "It would be great to have a dark theme option so users can switch UI themes at night."
    if b3.button("📚 Sample Documentation", use_container_width=True):
        st.session_state["title"] = "Update API installation guide for Python 3.12"
        st.session_state["body"] = "The README setup instructions are outdated and missing the required pip install flags."

    title_input = st.text_input("Issue Title", value=st.session_state.get("title", ""), placeholder="e.g., App crashes on clicking submit")
    body_input = st.text_area("Issue Description", value=st.session_state.get("body", ""), height=130, placeholder="Paste steps to reproduce, error logs, or feature details...")

    if st.button("🔍 Analyze Issue", type="primary"):
        if not title_input.strip() and not body_input.strip():
            st.warning("Please enter an issue title or description.")
        else:
            cleaned = clean_github_text(f"{title_input} {body_input}")
            pred_label, conf, probs = predict_issue(cleaned, model_choice, tokenizer, bert_model, baseline_model)

            q_emb = sbert_model.encode([cleaned], normalize_embeddings=True)
            scores = cosine_similarity(q_emb, ref_embeddings)[0]
            top3_idx = np.argsort(scores)[::-1][:3]
            top_score = float(scores[top3_idx[0]])

            st.markdown("---")
            left, right = st.columns([1, 1.35], gap="large")

            with left:
                st.markdown("#### 🏷️ Predicted Category")
                border_col, bg_col, badge_text = BADGE_STYLE[pred_label]
                st.markdown(
                    f'<div class="result-box">'
                    f'<span class="pill-badge" style="color:{border_col}; background:{bg_col}; border:1px solid {border_col};">{badge_text} · {conf*100:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown("<br>", unsafe_allow_html=True)
                st.write("**Class Confidence Breakdown**")
                for idx, lbl in ID2LABEL.items():
                    st.progress(float(probs[idx]), text=f"{lbl}: {probs[idx]*100:.1f}%")

            with right:
                st.markdown("#### 🔎 Top-3 Similar Existing Issues")
                if top_score >= dup_threshold:
                    st.error(f"⚠️ **Potential Duplicate Detected** — Top match is **{top_score*100:.1f}%** (Threshold: {dup_threshold*100:.0f}%)")
                else:
                    st.success(f"✅ **Likely Unique** — Highest match is **{top_score*100:.1f}%** (Threshold: {dup_threshold*100:.0f}%)")

                for rank, idx in enumerate(top3_idx, 1):
                    sim = float(scores[idx]) * 100
                    row = ref_df.iloc[idx]
                    with st.expander(f"#{rank} · {sim:.1f}% Match · ({row['clean_label']}) {str(row['title'])[:80]}"):
                        st.write(str(row["body"])[:400] + "...")

with tab2:
    st.caption("Pull real-time open issues from any public GitHub repository and classify them live.")
    c1, c2 = st.columns([3, 1])
    repo_input = c1.text_input("Public GitHub Repository (`owner/repo`)", value="streamlit/streamlit", placeholder="e.g., pallets/flask")
    issue_limit = c2.slider("Issues to Fetch", 5, 25, 10)

    if st.button("🚀 Fetch & Predict Live Issues", type="primary"):
        with st.spinner(f"Fetching live issues from {repo_input}..."):
            live_issues, err = fetch_live_github_issues(repo_input, issue_limit)

        if err:
            st.error(err)
        elif not live_issues:
            st.warning("No open issues found in this repository.")
        else:
            cleaned_texts = [clean_github_text(f"{i['title']} {i.get('body') or ''}") for i in live_issues]
            live_embeddings = sbert_model.encode(cleaned_texts, normalize_embeddings=True)
            sim_matrix = cosine_similarity(live_embeddings)

            rows = []
            for idx, issue in enumerate(live_issues):
                pred_label, conf, _ = predict_issue(cleaned_texts[idx], model_choice, tokenizer, bert_model, baseline_model)
                sims = sim_matrix[idx].copy()
                sims[idx] = -1.0
                best_match_idx = int(np.argmax(sims))
                best_sim = float(sims[best_match_idx])
                dup_flag = f"⚠️ #{live_issues[best_match_idx]['number']} ({best_sim*100:.0f}%)" if best_sim >= dup_threshold else "✅ Unique"

                rows.append({
                    "Issue #": f"#{issue['number']}",
                    "Title": issue["title"][:80],
                    "Predicted Label": BADGE_STYLE[pred_label][2],
                    "Confidence": f"{conf*100:.1f}%",
                    "Live Duplicate Check": dup_flag,
                    "Link": issue["html_url"],
                })

            live_df = pd.DataFrame(rows)
            st.success(f"Analyzed **{len(live_df)} live open issues** from `{repo_input}`.")
            st.dataframe(
                live_df,
                use_container_width=True,
                column_config={"Link": st.column_config.LinkColumn("GitHub URL", display_text="Open Issue")},
                hide_index=True,
            )