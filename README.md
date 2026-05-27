---
title: CSVmyPDFV1.0
emoji: 📄
colorFrom: gray
colorTo: gray
sdk: docker
app_file: app.py
pinned: false
---


# CSVmyPDF v1.0

Convert your bank statement PDFs (digital or scanned) into clean, structured CSVs with a single click.

---

## Running

The Hugging Face Space is deployed and can be used as is to check that the product works.
No API key is needed for that. The `GROQ_API_KEY` is only needed for local development or extensions.

**Optional local setup:**

1. **Clone the repo:**
    ```bash
    git clone https://github.com/Daniyalsid26/CSVmyPDFV1.0.git
    cd CSVmyPDFV1.0
    ```
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Set your API key for local use:**
    ```bash
    export GROQ_API_KEY=your_key
    ```
4. **Run the app:**
    ```bash
    python app.py
    ```

**Or with Docker:**
```bash
    docker build -t csvmypdf .
    docker run -p 7860:7860 -e GROQ_API_KEY=your_key csvmypdf
```

## Features
- **Drag-and-drop UI** - Upload one or more PDFs, get instant CSV downloads.
- **ZIP download button** - Download multiple CSVs together in one file.
- **Two-stage extraction** - Fast deterministic table parser for digital PDFs; LLM fallback for messy or scanned statements.
- **Automatic OCR** - Scanned PDFs are routed through Tesseract OCR for robust extraction.
- **Privacy-first** - No data is stored server-side. All processing is in-memory and ephemeral.
- **No vendor lock-in** - Runs locally or on Hugging Face Spaces. No proprietary formats.
- **Transparent cost** - LLM calls (Groq) are only used when necessary, minimizing API usage and cost.

---

## Architecture

```
PDF(s)
 │
 ├─ Stage 1: Table Extraction (pdfplumber)
 │    • Attempts to parse column-aligned tables directly from PDF.
 │    • If successful, skips LLM entirely (fast, free, deterministic).
 │
 └─ Stage 2: LLM Extraction (Groq Llama-3-70B)
        • If no table found, extracts raw text (OCR if needed).
        • Sends text to LLM with structured prompt for robust parsing.
        • Normalizes dates, amounts, and infers payment types.

Output: CSV(s) with columns: `date`, `payment_type`, `details`, `paid_out`, `paid_in`, `balance`
```

**Key modules:**
| File           | Role                                                      |
|----------------|-----------------------------------------------------------|
| `app.py`       | Gradio UI, custom CSS, file upload/download, footer       |
| `pipeline.py`  | Orchestrates extraction, normalization, CSV writing       |
| `extraction.py`| PDF text/OCR extraction, PII redaction, table parsing     |
| `llm.py`       | Groq LLM client, system/table prompts, extraction logic   |
| `models.py`    | Pydantic models, normalization, payment type inference    |

---

## Assumptions and limitations

The workflow assumes common bank statement layouts and keeps the main path simple:

- Table parsing is tried first, with LLM fallback only when needed.
- The app uses basic file validation and fail-soft recovery to keep processing stable.
- Long statements may be truncated, OCR quality depends on the scan, and LLM output should still be reviewed.

---

## Privacy & Security
- **No data retention:** Uploaded PDFs are never stored or logged.
- **Ephemeral processing:** All files are processed in-memory and deleted after conversion.
- **Open source:** Review the code, run locally, or deploy on your own infrastructure.

---

## Cost & API Key
- **LLM usage:** Only ambiguous or scanned statements are sent to Groq LLM. Digital PDFs with clean tables are processed locally (no API call).
- **Minimized cost:** The pipeline is designed to avoid unnecessary LLM calls, keeping your API usage low.
- **Local development:** Use `GROQ_API_KEY` only when running locally or making extensions.

---

## Confidence Score
After each conversion, a confidence score is shown. It reflects extraction trustworthiness based on:
- Extraction method (table vs LLM)
- Date quality (≥90% parse cleanly)
- Amount quality (≥90% rows have monetary value)
- Balance continuity (row-to-row check)

**Score meanings:**
- **≥ 80%** - Output is reliable
- **60-79%** - Minor issues, spot-check recommended
- **< 60%** - Significant issues, manual review required

---

---

## Contributing & License
- PRs welcome! Please add clear comments and update the README for major changes.
- MIT License.

---

## Contact
- [Daniyal Siddiqui](mailto:daniyal.siddiqui@finalto.com)
- [GitHub](https://github.com/Daniyalsid26/CSVmyPDFV1.0)
- [Hugging Face Space](https://huggingface.co/spaces/DaniyalSid/CSVmyPDFV1.0)
