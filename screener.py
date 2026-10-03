"""Resume Screener: rank resumes against a job description.

Final score = weighted mix of
  1. Keyword match   - share of must-have keywords found in the resume
  2. TF-IDF cosine   - how close the resume's wording is to the job description
"""
import re
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SUPPORTED = {".pdf", ".docx", ".txt"}


# ---------- Step 1: read resumes ----------
def extract_text(file, filename):
    """Return plain text from a PDF, DOCX or TXT (a path or an uploaded file object)."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        import pdfplumber

        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    if ext == ".docx":
        from docx import Document

        return "\n".join(p.text for p in Document(file).paragraphs)
    if ext == ".txt":
        if hasattr(file, "read"):
            data = file.read()
            return data.decode("utf-8", errors="ignore") if isinstance(data, bytes) else data
        return Path(file).read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {ext}")


# ---------- Step 2: clean text ----------
def clean_text(text):
    """Lowercase, drop punctuation (keeps + and # for things like c++ / c#), squeeze spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------- Step 3: keyword matching ----------
def keyword_match(resume_text, keywords):
    """Return (matched, missing, fraction_matched) for the must-have keywords."""
    text = clean_text(resume_text)
    matched, missing = [], []
    for kw in keywords:
        k = clean_text(kw)
        if k and re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", text):
            matched.append(kw)
        else:
            missing.append(kw)
    fraction = len(matched) / len(keywords) if keywords else 0.0
    return matched, missing, fraction


# ---------- Step 4: similarity to the job description ----------
def similarity_scores(job_description, resume_texts):
    """TF-IDF cosine similarity of each resume to the job description (0 to 1)."""
    docs = [clean_text(job_description)] + [clean_text(t) for t in resume_texts]
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(docs)
    return cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()


# ---------- Step 5: score and rank ----------
def rank_candidates(resumes, job_description, keywords, keyword_weight=0.6):
    """resumes: {filename: text}. Returns a list of dicts, best candidate first."""
    keywords = [k.strip() for k in keywords if k.strip()]
    if not keywords:
        keyword_weight = 0.0
    names = list(resumes)
    sims = similarity_scores(job_description, [resumes[n] for n in names])

    results = []
    for name, sim in zip(names, sims):
        matched, missing, kw_fraction = keyword_match(resumes[name], keywords)
        score = keyword_weight * kw_fraction + (1 - keyword_weight) * float(sim)
        results.append(
            {
                "name": name,
                "score": score,
                "keyword_score": kw_fraction,
                "similarity": float(sim),
                "matched": matched,
                "missing": missing,
            }
        )
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ---------- Quick command-line test ----------
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('Usage: python screener.py <resumes_folder> <job_description.txt> "kw1, kw2, kw3"')
        sys.exit(1)

    folder, jd_path, kw_arg = sys.argv[1:]
    resumes = {
        p.name: extract_text(str(p), p.name)
        for p in sorted(Path(folder).iterdir())
        if p.suffix.lower() in SUPPORTED
    }
    jd = Path(jd_path).read_text(encoding="utf-8")
    ranked = rank_candidates(resumes, jd, kw_arg.split(","))

    for i, r in enumerate(ranked, 1):
        print(f"{i}. {r['name']}  score={r['score'] * 100:.1f}%  "
              f"(keywords {r['keyword_score'] * 100:.0f}%, similarity {r['similarity'] * 100:.0f}%)")
        print(f"   matched: {', '.join(r['matched']) or '-'}")
        print(f"   missing: {', '.join(r['missing']) or '-'}")
