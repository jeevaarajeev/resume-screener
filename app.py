import pandas as pd
import streamlit as st

from screener import extract_text, rank_candidates

st.set_page_config(page_title="Resume Screener", page_icon="📄", layout="wide")
st.title("📄 Resume Screener")
st.caption("Upload resumes, paste a job description, and get a ranked shortlist.")

with st.sidebar:
    st.header("Settings")
    weight = st.slider(
        "Keyword weight",
        0.0, 1.0, 0.6, 0.05,
        help="Higher = must-have keywords matter more. Lower = overall similarity to the job description matters more.",
    )

jd = st.text_area("Job description", height=200, placeholder="Paste the job description here...")
kw_input = st.text_input(
    "Must-have keywords (comma-separated)",
    "recruitment, resume screening, talent management, communication",
)
files = st.file_uploader(
    "Upload resumes (PDF, DOCX or TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

if st.button("Rank candidates", type="primary"):
    if not jd.strip() or not files:
        st.warning("Please paste a job description and upload at least one resume.")
        st.stop()

    resumes = {}
    for f in files:
        try:
            text = extract_text(f, f.name)
        except Exception as e:
            st.error(f"Could not read {f.name}: {e}")
            continue
        if not text.strip():
            st.warning(f"{f.name} has no readable text (it may be a scanned image).")
            continue
        resumes[f.name] = text

    if not resumes:
        st.stop()

    keywords = [k.strip() for k in kw_input.split(",") if k.strip()]
    results = rank_candidates(resumes, jd, keywords, weight)

    df = pd.DataFrame(
        [
            {
                "Rank": i,
                "Candidate": r["name"],
                "Score (%)": round(r["score"] * 100, 1),
                "Keyword match (%)": round(r["keyword_score"] * 100),
                "Similarity (%)": round(r["similarity"] * 100),
                "Matched": ", ".join(r["matched"]),
                "Missing": ", ".join(r["missing"]),
            }
            for i, r in enumerate(results, 1)
        ]
    )

    st.subheader("Ranking")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("Candidate")["Score (%)"])
    st.download_button("Download ranking as CSV", df.to_csv(index=False), "ranking.csv", "text/csv")
