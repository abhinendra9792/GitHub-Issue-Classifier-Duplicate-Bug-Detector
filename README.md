# 🛠️ GitHub Issue Classifier & Duplicate Bug Detector (NLP-46)

An end-to-end Natural Language Processing (NLP) web application that automates GitHub issue triage by classifying issues into **Bug**, **Feature Request**, or **Documentation**, assigning **auto-priority tags**, and detecting **semantic duplicates** in real time.

- **Live Demo:** `<YOUR-STREAMLIT-URL>`
- **Fine-Tuned Model (Hugging Face):** [abhinendra9792/distilbert-github-issue-classifier](https://huggingface.co/abhinendra9792/distilbert-github-issue-classifier)
- **Course:** CSB306 — Natural Language Processing

---

## ✨ Key Features

1. **Manual Issue Triage & Model Comparison:** Classify custom or sample issues using Fine-Tuned DistilBERT, TF-IDF Baseline, or **Side-by-Side Comparison Mode**, complete with **Auto-Severity/Priority tagging** and **TF-IDF keyword explainability**.
2. **Real-Time GitHub Repo Triage:** Fetches live open issues from any public GitHub repository via the GitHub REST API, classifies them live, detects intra-repo duplicates, plots a category breakdown chart, and exports a `.csv` triage report.
3. **Semantic Duplicate Detection:** Uses Sentence-BERT (`all-MiniLM-L6-v2`) with 384-dimensional embeddings and cosine similarity against a 2,000-issue reference index.
4. **Batch CSV Upload:** Upload a CSV file with `title` and `body` columns to auto-label up to 100 issues in one click.
5. **Model Evaluation Dashboard:** Displays benchmark metrics, dataset class distribution, DistilBERT confusion matrix, and qualitative error analysis.
6. **SQLite Database Logging (`database.py`):** Automatically logs every prediction into `triage_history.db` (`triage_logs` table) with filtering, CSV export, and history management in the **Triage History (DB)** tab.

---

## 📊 Model Performance (Test Set — 1,350 Balanced Issues)

| Model | Type | Test Accuracy | Macro F1 |
| :--- | :--- | :---: | :---: |
| TF-IDF + Logistic Regression | Classical NLP Baseline | 69.85% | 0.6976 |
| **DistilBERT (Fine-Tuned)** | **Transformer (Deep Learning)** | **75.85%** | **0.7579** |

> Fine-tuning DistilBERT delivered a **+6.03% Macro F1** improvement over the classical TF-IDF baseline.

---

## 🗂️ Project Structure

```text
├── app.py                             # 5-tab Streamlit web application
├── database.py                        # SQLite database initialization & logging (triage_logs)
├── requirements.txt                   # Python dependencies
├── tfidf_baseline.joblib              # Trained TF-IDF + Logistic Regression pipeline
├── reference_issues.csv               # 2,000 reference GitHub issues for duplicate search
├── issue_embeddings.npy               # Precomputed SBERT embeddings (2000 x 384)
├── class_distribution.png             # Dataset balance visualization
└── confusion_matrix_distilbert.png    # DistilBERT test-set confusion matrix