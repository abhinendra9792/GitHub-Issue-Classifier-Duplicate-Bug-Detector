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
HIGH_PRIORITY_TERMS = {
    "crash", "crashes", "crashing", "security", "vulnerability", "exploit",
    "data loss", "corrupt", "corruption", "memory leak", "freeze", "freezes",
    "deadlock", "nullpointerexception", "segfault", "outage", "critical", "fatal"
}
MEDIUM_PRIORITY_TERMS = {
    "error", "exception", "fail", "fails", "failed", "broken", "bug",
    "incorrect", "wrong", "slow", "timeout", "hang", "regression"
}

st.set_page_config(page_title="GitHub Issue Triage (NLP-46)", page_icon="🛠️", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    [data-testid="stToolbar"] { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .hero-card {
        background: linear-gradient(135deg, #161b22 0%, #1f2937 55%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 2.6rem !important;
        font-weight: 800;
        color: #f0f6fc;
        margin: 0;
        line-height: 1.2;
    }
    .kpi-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: left;
    }
    .kpi-label { color: #8b949e; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { color: #f0f6fc; font-size: 1.5rem; font-weight: 700; margin-top: 0.15rem; }
    .kpi-delta { color: #3fb950; font-size: 0.82rem; font-weight: 600; }
    .pill-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 1rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .result-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HELPER FUNCTIONS ----------------
def clean_github_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"```[\s\S]*?```", " [CODE] ", text)
    text = re.sub(r"`[^`]*`", " [CODE] ", text)
    text = re.sub(r"http\S+|www\.\S+", " [URL] ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[#*_>\[\]\(\)]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def get_priority_tag(pred_label: str, raw_text: str):
    lower = raw_text.lower()
    if pred_label == "Bug":
        if any(term in lower for term in HIGH_PRIORITY_TERMS):
            return "🔴 High Priority", "#ff4b4b", "rgba(255, 75, 75, 0.15)"
        if any(term in lower for term in MEDIUM_PRIORITY_TERMS):
            return "🟠 Medium Priority", "#d29922", "rgba(210, 153, 34, 0.15)"
        return "🟡 Normal Priority", "#e3b341", "rgba(227, 179, 65, 0.15)"
    if pred_label == "Feature Request":
        return "🟢 Enhancement", "#2ea043", "rgba(46, 160, 67, 0.15)"
    return "🔵 Low Priority (Docs)", "#388bfd", "rgba(56, 139, 253, 0.15)"

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

def predict_distilbert(cleaned_text: str, tokenizer, bert_model):
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        probs = torch.softmax(bert_model(**inputs).logits, dim=-1).numpy()[0]
    pred_idx = int(np.argmax(probs))
    return ID2LABEL[pred_idx], float(probs[pred_idx]), probs

def predict_tfidf(cleaned_text: str, baseline_model):
    probs = baseline_model.predict_proba([cleaned_text])[0]
    pred_idx = int(np.argmax(probs))
    return ID2LABEL[pred_idx], float(probs[pred_idx]), probs

def get_top_tfidf_keywords(cleaned_text: str, baseline_model, pred_idx: int, top_n: int = 6):
    try:
        vectorizer = baseline_model.steps[0][1]
        classifier = baseline_model.steps[-1][1]
        vec = vectorizer.transform([cleaned_text])
        feature_names = vectorizer.get_feature_names_out()
        coefs = classifier.coef_[pred_idx]
        scores = vec.toarray()[0] * coefs
        nonzero = np.where(vec.toarray()[0] > 0)[0]
        ranked = sorted(nonzero, key=lambda i: scores[i], reverse=True)
        return [(feature_names[i], float(scores[i])) for i in ranked[:top_n] if scores[i] > 0]
    except Exception:
        return []

@st.cache_data(ttl=300)
def fetch_live_github_issues(repo_name: str, count: int = 10):
    url = f"https://api.github.com/repos/{repo_name.strip()}/issues"
    params = {"state": "open", "per_page": 100}
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
        [
            "DistilBERT (Fine-Tuned)",
            "TF-IDF + Logistic Regression (Baseline)",
            "⚔️ Compare Both Models"
        ]
    )
    dup_threshold = st.slider("Duplicate Similarity Threshold", 0.50, 0.95, 0.70, 0.05)
    st.markdown("---")
    st.markdown("**Tech Stack**")
    st.caption(
        "• Classifier: Fine-tuned `distilbert-base-uncased`\n"
        "• Baseline: TF-IDF + Logistic Regression\n"
        "• Duplicates: `all-MiniLM-L6-v2` (SBERT) + Cosine Similarity\n"
        "• Live Data: GitHub REST API"
    )

# ---------------- HERO BANNER ----------------
st.markdown("""
<div class="hero-card">
    <h1 class="hero-title">🛠️ GitHub Issue Classifier & Duplicate Bug Detector</h1>
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
tab1, tab2, tab3, tab4 = st.tabs([
    "✍️ Manual Issue Triage",
    "🔴 Live GitHub Repo Triage",
    "📂 Batch CSV Upload",
    "📊 Model Evaluation"
])

# ---------------- TAB 1: MANUAL INPUT ----------------
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
            raw_text = f"{title_input} {body_input}"
            cleaned = clean_github_text(raw_text)

            # Duplicate Detection (SBERT)
            q_emb = sbert_model.encode([cleaned], normalize_embeddings=True)
            scores = cosine_similarity(q_emb, ref_embeddings)[0]
            top3_idx = np.argsort(scores)[::-1][:3]
            top_score = float(scores[top3_idx[0]])

            st.markdown("---")

            if model_choice.startswith("⚔️"):
                b_label, b_conf, b_probs = predict_distilbert(cleaned, tokenizer, bert_model)
                t_label, t_conf, t_probs = predict_tfidf(cleaned, baseline_model)
                p_text, p_col, p_bg = get_priority_tag(b_label, raw_text)

                st.markdown("#### ⚔️ Side-by-Side Model Comparison")
                mc1, mc2 = st.columns(2)
                with mc1:
                    bc, bg, bt = BADGE_STYLE[b_label]
                    st.markdown(
                        f'<div class="result-box"><strong>DistilBERT (Fine-Tuned)</strong><br><br>'
                        f'<span class="pill-badge" style="color:{bc}; background:{bg}; border:1px solid {bc};">{bt} · {b_conf*100:.1f}%</span>'
                        f'<span class="pill-badge" style="color:{p_col}; background:{p_bg}; border:1px solid {p_col};">{p_text}</span></div>',
                        unsafe_allow_html=True
                    )
                    for idx, lbl in ID2LABEL.items():
                        st.progress(float(b_probs[idx]), text=f"{lbl}: {b_probs[idx]*100:.1f}%")

                with mc2:
                    tc, tbg, tt = BADGE_STYLE[t_label]
                    st.markdown(
                        f'<div class="result-box"><strong>TF-IDF + Logistic Regression (Baseline)</strong><br><br>'
                        f'<span class="pill-badge" style="color:{tc}; background:{tbg}; border:1px solid {tc};">{tt} · {t_conf*100:.1f}%</span></div>',
                        unsafe_allow_html=True
                    )
                    for idx, lbl in ID2LABEL.items():
                        st.progress(float(t_probs[idx]), text=f"{lbl}: {t_probs[idx]*100:.1f}%")

                pred_idx_for_explain = int(np.argmax(t_probs))
            else:
                if model_choice.startswith("DistilBERT"):
                    pred_label, conf, probs = predict_distilbert(cleaned, tokenizer, bert_model)
                else:
                    pred_label, conf, probs = predict_tfidf(cleaned, baseline_model)

                p_text, p_col, p_bg = get_priority_tag(pred_label, raw_text)
                left, right = st.columns([1, 1.35], gap="large")

                with left:
                    st.markdown("#### 🏷️ Predicted Category & Priority")
                    bc, bg, bt = BADGE_STYLE[pred_label]
                    st.markdown(
                        f'<div class="result-box">'
                        f'<span class="pill-badge" style="color:{bc}; background:{bg}; border:1px solid {bc};">{bt} · {conf*100:.1f}%</span>'
                        f'<span class="pill-badge" style="color:{p_col}; background:{p_bg}; border:1px solid {p_col};">{p_text}</span>'
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

                pred_idx_for_explain = int(np.argmax(probs))

            # If Compare Mode, show duplicates below the comparison
            if model_choice.startswith("⚔️"):
                st.markdown("---")
                st.markdown("#### 🔎 Top-3 Similar Existing Issues (SBERT)")
                if top_score >= dup_threshold:
                    st.error(f"⚠️ **Potential Duplicate Detected** — Top match is **{top_score*100:.1f}%** (Threshold: {dup_threshold*100:.0f}%)")
                else:
                    st.success(f"✅ **Likely Unique** — Highest match is **{top_score*100:.1f}%** (Threshold: {dup_threshold*100:.0f}%)")
                for rank, idx in enumerate(top3_idx, 1):
                    sim = float(scores[idx]) * 100
                    row = ref_df.iloc[idx]
                    with st.expander(f"#{rank} · {sim:.1f}% Match · ({row['clean_label']}) {str(row['title'])[:80]}"):
                        st.write(str(row["body"])[:400] + "...")

            # Preprocessing & Explainability Expander
            with st.expander("🧠 Show NLP Pipeline & Keyword Explainability"):
                st.markdown("**1. Cleaned Text (After Regex Preprocessing):**")
                st.code(cleaned, language="text")
                top_words = get_top_tfidf_keywords(cleaned, baseline_model, pred_idx_for_explain)
                st.markdown("**2. Top Influential Keywords (TF-IDF Feature Weights):**")
                if top_words:
                    kw_str = " · ".join([f"`{w}` ({s:.3f})" for w, s in top_words])
                    st.markdown(kw_str)
                else:
                    st.caption("No strong TF-IDF keyword weights found for this short input.")

# ---------------- TAB 2: LIVE GITHUB REPO ----------------
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
            raw_texts = [f"{i['title']} {i.get('body') or ''}" for i in live_issues]
            cleaned_texts = [clean_github_text(t) for t in raw_texts]
            live_embeddings = sbert_model.encode(cleaned_texts, normalize_embeddings=True)
            sim_matrix = cosine_similarity(live_embeddings)

            rows = []
            label_counts = {"Bug": 0, "Feature Request": 0, "Documentation": 0}
            for idx, issue in enumerate(live_issues):
                if model_choice.startswith("TF-IDF"):
                    pred_label, conf, _ = predict_tfidf(cleaned_texts[idx], baseline_model)
                else:
                    pred_label, conf, _ = predict_distilbert(cleaned_texts[idx], tokenizer, bert_model)

                label_counts[pred_label] += 1
                priority_text, _, _ = get_priority_tag(pred_label, raw_texts[idx])

                sims = sim_matrix[idx].copy()
                sims[idx] = -1.0
                best_match_idx = int(np.argmax(sims))
                best_sim = float(sims[best_match_idx])
                dup_flag = f"⚠️ #{live_issues[best_match_idx]['number']} ({best_sim*100:.0f}%)" if best_sim >= dup_threshold else "✅ Unique"

                rows.append({
                    "Issue #": f"#{issue['number']}",
                    "Title": issue["title"][:80],
                    "Predicted Label": BADGE_STYLE[pred_label][2],
                    "Priority": priority_text,
                    "Confidence": f"{conf*100:.1f}%",
                    "Live Duplicate Check": dup_flag,
                    "Link": issue["html_url"],
                })

            live_df = pd.DataFrame(rows)
            st.success(f"Analyzed **{len(live_df)} live open issues** from `{repo_input}`.")

            # Live Repo Analytics Chart
            st.markdown("#### 📊 Live Category Breakdown")
            chart_df = pd.DataFrame.from_dict(label_counts, orient="index", columns=["Issues"])
            st.bar_chart(chart_df)

            st.dataframe(
                live_df,
                use_container_width=True,
                column_config={"Link": st.column_config.LinkColumn("GitHub URL", display_text="Open Issue")},
                hide_index=True,
            )

            csv_bytes = live_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Triage Report (.csv)",
                data=csv_bytes,
                file_name=f"{repo_input.replace('/', '_')}_triage_report.csv",
                mime="text/csv"
            )

# ---------------- TAB 3: BATCH CSV UPLOAD ----------------
with tab3:
    st.caption("Upload a CSV file containing `title` and `body` columns to classify all issues in one click.")
    uploaded_file = st.file_uploader("Upload Issues CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        if "title" not in batch_df.columns:
            st.error("Your CSV must contain at least a `title` column (and optionally a `body` column).")
        else:
            st.info(f"Loaded **{len(batch_df)} rows**. Preview:")
            st.dataframe(batch_df.head(5), use_container_width=True, hide_index=True)

            max_rows = min(len(batch_df), 100)
            if len(batch_df) > 100:
                st.warning("For fast inference on Streamlit Cloud, batch mode processes the first 100 rows.")

            if st.button("⚡ Run Batch Classification", type="primary"):
                sub_df = batch_df.head(max_rows).copy()
                preds, confs, priorities = [], [], []
                progress = st.progress(0)

                for i, row in enumerate(sub_df.itertuples(index=False)):
                    t = getattr(row, "title", "")
                    b = getattr(row, "body", "") if "body" in sub_df.columns else ""
                    raw = f"{t} {b}"
                    cleaned = clean_github_text(raw)
                    if model_choice.startswith("TF-IDF"):
                        lbl, cf, _ = predict_tfidf(cleaned, baseline_model)
                    else:
                        lbl, cf, _ = predict_distilbert(cleaned, tokenizer, bert_model)
                    prio, _, _ = get_priority_tag(lbl, raw)
                    preds.append(lbl)
                    confs.append(f"{cf*100:.1f}%")
                    priorities.append(prio)
                    progress.progress((i + 1) / len(sub_df))

                sub_df["predicted_label"] = preds
                sub_df["confidence"] = confs
                sub_df["priority"] = priorities

                st.success(f"Classified **{len(sub_df)} issues** successfully.")
                st.dataframe(sub_df, use_container_width=True, hide_index=True)
                st.download_button(
                    label="⬇️ Download Labeled CSV",
                    data=sub_df.to_csv(index=False).encode("utf-8"),
                    file_name="batch_triaged_issues.csv",
                    mime="text/csv"
                )

# ---------------- TAB 4: MODEL EVALUATION ----------------
with tab4:
    st.markdown("#### 📈 Benchmark Comparison (Test Set — 1,350 Balanced Issues)")
    eval_df = pd.DataFrame([
        {"Model": "TF-IDF + Logistic Regression (Baseline)", "Test Accuracy": "69.85%", "Macro F1": "0.6976", "Type": "Classical NLP"},
        {"Model": "DistilBERT (Fine-Tuned)", "Test Accuracy": "75.85%", "Macro F1": "0.7579", "Type": "Transformer (Deep Learning)"},
    ])
    st.dataframe(eval_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    c_img1, c_img2 = st.columns(2)
    with c_img1:
        st.markdown("#### Dataset Class Distribution")
        st.image("class_distribution.png", use_container_width=True)
    with c_img2:
        st.markdown("#### DistilBERT Confusion Matrix")
        st.image("confusion_matrix_distilbert.png", use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔍 Qualitative Error Analysis (5 Sample Misclassifications)")
    error_df = pd.DataFrame([
        {"Issue Title": "test fractal pattern library with windows users", "Actual": "Documentation", "Predicted": "Feature Request", "Why It Happened": "Action-oriented testing request phrased like a new feature."},
        {"Issue Title": "save avatar image on social login", "Actual": "Bug", "Predicted": "Feature Request", "Why It Happened": "Reads like a feature addition without explicit crash/error keywords."},
        {"Issue Title": "perimeter unglued them", "Actual": "Feature Request", "Predicted": "Documentation", "Why It Happened": "Body contained mostly template version instructions and URLs."},
        {"Issue Title": "enable to get commits of a branch", "Actual": "Documentation", "Predicted": "Feature Request", "Why It Happened": "Starts with 'enable to get...', a strong Feature Request phrasing."},
        {"Issue Title": "desired fixes in inbo_rapport_2015: white space in section titles", "Actual": "Feature Request", "Predicted": "Bug", "Why It Happened": "Word 'fixes' strongly signals Bug to the classifier."},
    ])
    st.dataframe(error_df, use_container_width=True, hide_index=True)