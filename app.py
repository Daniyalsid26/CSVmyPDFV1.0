import gradio as gr
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
import csv
import json
import os
import uuid
from groq import AsyncGroq
from pydantic import BaseModel, Field
from typing import List, Optional


# ─── Pydantic Schema ──────────────────────────────────────────────────────────

class Transaction(BaseModel):
    date: Optional[str] = Field(
        None,
        description="Transaction date, normalised to yyyy-mm-dd where possible. Leave blank if not available.",
    )
    payment_type: Optional[str] = Field(
        None,
        description=(
            "Type of transaction, such as card payment, transfer, direct debit, "
            "standing order, fee, interest, salary, or cash withdrawal. Leave blank if not available."
        ),
    )
    details: Optional[str] = Field(
        None,
        description="Cleaned transaction description or narrative from the statement. Leave blank if not available.",
    )
    paid_out: Optional[float] = Field(
        None,
        description="Outgoing payment amount (debit / withdrawal / charge). Leave blank if not available.",
    )
    paid_in: Optional[float] = Field(
        None,
        description="Incoming payment amount (credit / deposit / receipt). Leave blank if not available.",
    )
    balance: Optional[float] = Field(
        None,
        description="Running balance after the transaction. Leave blank if not available.",
    )


class BankStatement(BaseModel):
    transactions: List[Transaction] = Field(
        description="List of all chronological transactions found in the document."
    )


# ─── PII Redaction ────────────────────────────────────────────────────────────
# Strip only high-confidence standalone identifiers; leave narrative text intact
# so the LLM can still interpret transaction descriptions accurately.

_PII_PATTERNS = [
    # 16-digit card numbers (with optional spaces or dashes between groups)
    (re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"), "[CARD]"),
    # IBAN: 2 uppercase letters + 2 digits + 11-30 alphanumeric chars
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), "[IBAN]"),
    # UK sort codes in XX-XX-XX format
    (re.compile(r"\b\d{2}-\d{2}-\d{2}\b"), "[SORT-CODE]"),
    # Standalone 8-digit account numbers (not part of a longer number sequence)
    (re.compile(r"(?<!\d)\d{8}(?!\d)"), "[ACCT]"),
]


def redact_pii(text: str) -> str:
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ─── PDF Text Extraction ──────────────────────────────────────────────────────

def extract_text(pdf_path: str) -> str:
    """
    Iterate through PDF pages. For pages with substantial digital text, use
    PyMuPDF's native extraction. For near-blank pages (scanned images), fall
    back to Tesseract OCR.
    """
    doc = fitz.open(pdf_path)
    pages: list[str] = []

    for page in doc:
        text = page.get_text()
        if len(text.strip()) < 50:
            # Scanned page — render to image and OCR
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)
        pages.append(text)

    doc.close()
    return "\n".join(pages)


# ─── LLM Extraction ───────────────────────────────────────────────────────────

client = AsyncGroq()  # Reads GROQ_API_KEY from environment automatically

_SYSTEM_PROMPT = (
    "Extract all transactions from the bank statement below. "
    "Map debits/withdrawals/DR to paid_out, credits/deposits/CR to paid_in. "
    "Dates as yyyy-mm-dd. Null for missing fields. "
    'Return JSON: {"transactions":[{"date","payment_type","details","paid_out","paid_in","balance"}]}'
)


async def parse_statement(raw_text: str) -> BankStatement:
    # Hard cap to stay within the model's context window
    if len(raw_text) > 90_000:
        raw_text = raw_text[:90_000]

    response = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": raw_text},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    data = json.loads(response.choices[0].message.content)
    return BankStatement.model_validate(data)


# ─── CSV Generation ───────────────────────────────────────────────────────────

_HEADERS = ["date", "payment_type", "details", "paid_out", "paid_in", "balance"]


def write_csv(statement: BankStatement, out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_HEADERS)
        for t in statement.transactions:
            writer.writerow([
                t.date or "",
                t.payment_type or "",
                t.details or "",
                "" if t.paid_out is None else t.paid_out,
                "" if t.paid_in is None else t.paid_in,
                "" if t.balance is None else t.balance,
            ])


# ─── Main Pipeline ────────────────────────────────────────────────────────────

async def process_pdf(pdf_path: str | None):
    if pdf_path is None:
        return None, "Please upload a PDF file."

    # Use unique tmp names so concurrent uploads never collide
    run_id = uuid.uuid4().hex
    tmp_pdf = f"/tmp/stmt_{run_id}.pdf"
    tmp_csv = f"/tmp/out_{run_id}.csv"

    try:
        # Stage 1 — Ingest
        with open(pdf_path, "rb") as src, open(tmp_pdf, "wb") as dst:
            dst.write(src.read())

        # Stage 2 — Harvest text + OCR fallback + PII redaction
        raw_text = extract_text(tmp_pdf)
        clean_text = redact_pii(raw_text)

        # Stage 3 — LLM structured extraction
        statement = await parse_statement(clean_text)

        # Stage 4 — CSV output
        write_csv(statement, tmp_csv)

        count = len(statement.transactions)
        return tmp_csv, f"Done — {count} transaction{'s' if count != 1 else ''} extracted."

    except Exception as exc:
        return None, f"Error: {exc}"

    finally:
        if os.path.exists(tmp_pdf):
            os.remove(tmp_pdf)


# ─── Gradio UI ────────────────────────────────────────────────────────────────

_CSS = """
body, .gradio-container { font-family: 'Courier New', monospace !important; }
#title { text-align: center; }
"""

with gr.Blocks(css=_CSS, title="CSV my PDF") as demo:
    gr.Markdown("# CSV my PDF", elem_id="title")
    gr.Markdown(
        "Upload a bank statement PDF (digital or scanned). "
        "The AI extracts every transaction and returns a clean CSV."
    )

    pdf_input = gr.File(
        label="Bank Statement (PDF)",
        file_types=[".pdf"],
        type="filepath",
    )
    convert_btn = gr.Button("Convert to CSV", variant="primary")
    status_box = gr.Textbox(label="Status", interactive=False, lines=1)
    csv_output = gr.File(label="Download CSV")

    convert_btn.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=[csv_output, status_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
