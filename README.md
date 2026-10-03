# 📄 Resume Screener

A Python + NLP tool that ranks resumes against a job description. Upload PDF, DOCX or TXT resumes, paste a job description, list the must-have keywords, and get a ranked shortlist with matched and missing skills for each candidate.

## How it works

Each resume gets a score from two signals:

1. **Keyword match:** the share of must-have keywords found in the resume.
2. **TF-IDF cosine similarity:** how close the resume's wording is to the job description (scikit-learn).

`Final score = keyword_weight × keyword match + (1 − keyword_weight) × similarity`

The keyword weight (default 0.6) can be changed in the app.

## Tech stack

Python · scikit-learn · pdfplumber · python-docx · Streamlit · pandas

## Run it

```bash
git clone https://github.com/<your-username>/resume-screener.git
cd resume-screener
pip install -r requirements.txt

# Web app
streamlit run app.py

# Or command line
python screener.py sample_resumes sample_jd.txt "recruitment, resume screening, communication"
```

## Project structure

```
screener.py        core logic (extract, clean, match, score, rank)
app.py             Streamlit web interface
sample_jd.txt      example job description
sample_resumes/    example resumes for testing
```

## Limitations

- Scanned (image-only) PDFs can't be read without OCR.
- Keyword matching is literal, so "HR" and "human resources" are treated as different words.
- Scores support a human reviewer. They should not be the only basis for rejecting anyone.

## Future improvements

- Synonym matching with spaCy or sentence embeddings
- Automatic skill extraction from the job description
- OCR support for scanned resumes
- Export a shortlist email draft

## Author

Jeeva Rajeev · [LinkedIn](https://linkedin.com/in/your-link)
