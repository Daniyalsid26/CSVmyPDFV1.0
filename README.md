---
title: CSVmyPDFV1.0
emoji: 📄
colorFrom: gray
colorTo: gray
sdk: docker
app_file: app.py
pinned: false
---

# CSV my PDF V1.0

Upload bank statement PDFs (digital or scanned) and download clean, structured CSVs.

**Output columns:** `date` · `payment_type` · `details` · `paid_out` · `paid_in` · `balance`

---

## How to run

**Hugging Face Spaces** (deployed) — set `GROQ_API_KEY` as a Space secret, then use the UI.

**Locally:**
```bash
git clone https://github.com/Daniyalsid26/CSVmyPDFV1.0.git
cd CSVmyPDFV1.0
pip install -r requirements.txt
GROQ_API_KEY=your_key python app.py
```

**Docker:**
```bash
docker build -t csvmypdf .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key csvmypdf
```

---

## Architecture

```
PDF
 │
 ├─ Stage 1 (deterministic) — pdfplumber table extractor
 │   Detects and parses column-aligned tables directly.
 │   Skipped if no table structure is found.
 │
 └─ Stage 2 (LLM fallback) — Groq llama-3.3-70b-versatile
     Raw text is extracted (OCR via PyMuPDF + Tesseract for scanned PDFs)
     and sent to the LLM with a structured JSON prompt.
```

**Modules:**
| File | Role |
|---|---|
| `extraction.py` | PDF text extraction, OCR routing, PII redaction |
| `llm.py` | Groq client, system prompt, LLM parse call |
| `models.py` | Pydantic models, amount/date normalisation, payment type inference |
| `pipeline.py` | Orchestrator — runs extraction, normalises output, writes CSVs |
| `app.py` | Gradio UI |

---

## Design choices

- **Two-stage extraction** — table parser runs first; LLM is only used when no table structure is detected. Faster and cheaper for digital statements.
- **Document-level OCR routing** — if total extracted text across all pages is under 200 chars, the whole document is re-processed via OCR rather than routing page-by-page.
- **YYYY-MM-DD dates** — all date strings are normalised via `python-dateutil` so Excel opens them correctly.
- **Payment type inference** — 8-category regex rules applied post-extraction (`DIRECT_DEBIT`, `CARD_PAYMENT`, `TRANSFER`, etc.).
- **Named CSV output** — `Statement1.pdf` produces `Statement1.csv`; multiple files can be merged into `combined.csv` via checkbox.

---

## Limitations

- Long statements (100+ transactions) may be truncated — the LLM call is a single request with no chunking.
- OCR quality depends on scan resolution; results may vary on low-quality scans.
