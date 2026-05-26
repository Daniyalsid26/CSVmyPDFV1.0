"""pipeline.py — PDF extraction, table parsing, LLM fallback, CSV output."""
from __future__ import annotations

import csv
import io
import json
import os
import re
import uuid
from typing import Optional

import fitz  # PyMuPDF — used only for OCR page rendering
import pdfplumber
import pytesseract
from groq import AsyncGroq
from PIL import Image
from pydantic import BaseModel


# ─── Pydantic Models ──────────────────────────────────────────────────────────
# Two-tier: LLM/table parser returns strings; we parse amounts deterministically.

class RawTransaction(BaseModel):
    """Transaction as extracted — amounts kept as raw strings."""
    date: Optional[str] = None
    payment_type: Optional[str] = None
    details: Optional[str] = None
    paid_out: Optional[str] = None
    paid_in: Optional[str] = None
    balance: Optional[str] = None


class Transaction(BaseModel):
    """Normalised transaction with deterministically parsed float amounts."""
    date: Optional[str] = None
    payment_type: Optional[str] = None
    details: Optional[str] = None
    paid_out: Optional[float] = None
    paid_in: Optional[float] = None
    balance: Optional[float] = None


# ─── Amount Normalisation ─────────────────────────────────────────────────────

def normalize_amount(raw: Optional[str]) -> Optional[float]:
    """Parse a currency string to float.

    Handles: £/$, comma separators, CR/DR suffixes, parenthetical negatives,
    trailing/leading minus, European decimal format (1.234,56).
    """
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1].strip()
    elif text.endswith("-"):
        negative = True
        text = text[:-1].strip()
    elif text.startswith("-"):
        negative = True
        text = text[1:].strip()

    text = re.sub(r"[£$€¥\s]", "", text)
    text = re.sub(r"(?i)\s*(cr|dr)$", "", text).strip()

    if not text:
        return None

    # European decimal: 1.234,56
    if re.match(r"^\d{1,3}(\.\d{3})+,\d{1,2}$", text):
        text = text.replace(".", "").replace(",", ".")
    else:
        text = text.replace(",", "")

    try:
        value = float(text)
        return -value if negative else value
    except ValueError:
        return None


def normalize_raw(raw: RawTransaction) -> Transaction:
    """Convert a RawTransaction to a Transaction with parsed float amounts."""
    return Transaction(
        date=raw.date,
        payment_type=raw.payment_type,
        details=raw.details,
        paid_out=normalize_amount(raw.paid_out),
        paid_in=normalize_amount(raw.paid_in),
        balance=normalize_amount(raw.balance),
    )


# ─── PII Redaction ────────────────────────────────────────────────────────────

_PII_PATTERNS = [
    (re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"), "[CARD]"),
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), "[IBAN]"),
    (re.compile(r"\b\d{2}-\d{2}-\d{2}\b"), "[SORT-CODE]"),
    (re.compile(r"(?<!\d)\d{8}(?!\d)"), "[ACCT]"),
]


def redact_pii(text: str) -> str:
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ─── Deterministic Table Parser ───────────────────────────────────────────────
# Stage 1: use pdfplumber's embedded table extraction to read column cells
# directly, preserving paid_out/paid_in/balance column positions exactly.

# Header matching rules — more specific before broader
_HEADER_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bdate\b", re.I), "date"),
    (re.compile(
        r"\b(payment[\s_-]?type|transaction[\s_-]?type|tran[\s_-]?type|txn[\s_-]?type)\b",
        re.I,
    ), "payment_type"),
    (re.compile(r"\b(type|code|category)\b", re.I), "payment_type"),
    (re.compile(
        r"\b(description|details?|particulars|narrative|reference|memo|payee|remarks|narration)\b",
        re.I,
    ), "details"),
    # paid_out before paid_in so "out" doesn't shadow "in" match on same cell
    (re.compile(
        r"\b(debit|dr|withdrawals?|paid[\s_-]?out|money[\s_-]?out|charges?)\b",
        re.I,
    ), "paid_out"),
    (re.compile(
        r"\b(credit|cr|deposits?|paid[\s_-]?in|money[\s_-]?in|receipts?|income)\b",
        re.I,
    ), "paid_in"),
    (re.compile(r"\b(balance|bal)\b", re.I), "balance"),
]


def _match_header(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    h = re.sub(r"\s+", " ", raw.lower().strip())
    # Strip trailing currency decorations: (£) (GBP) etc.
    h = re.sub(r"\s*(?:\([£€$]\)|\([a-z]{3}\)|[£€$])\s*$", "", h).strip()
    for pattern, field in _HEADER_RULES:
        if pattern.search(h):
            return field
    return None


def _map_headers(row: list) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for idx, cell in enumerate(row):
        field = _match_header(str(cell) if cell is not None else None)
        if field and field not in mapping:
            mapping[field] = idx
    return mapping


def _cell(row: list, idx: Optional[int]) -> Optional[str]:
    if idx is None or idx >= len(row):
        return None
    val = row[idx]
    return str(val).strip() if val is not None else None


def _is_amount_like(value: Optional[str]) -> bool:
    if not value or not value.strip():
        return False
    clean = re.sub(r"[£€$¥,\s]", "", value.strip())
    clean = re.sub(r"(?i)(cr|dr)$", "", clean).strip()
    try:
        float(clean)
        return True
    except ValueError:
        return False


def _is_date_like(value: Optional[str]) -> bool:
    if not value:
        return False
    return bool(
        re.search(r"\d{1,4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,4}", value)
        or re.search(r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", value, re.I)
    )


def try_extract_tables(pdf_path: str) -> Optional[list[RawTransaction]]:
    """Attempt deterministic extraction from embedded PDF table structures.

    Reads column cells by index — column position is structurally guaranteed,
    so paid_out/paid_in can never be swapped.
    Returns None when no recognisable table structure is found, signalling the
    caller to fall through to the LLM path.
    """
    transactions: list[RawTransaction] = []
    found_structure = False
    # Carry the last known column mapping across pages so continuation tables
    # (page 2+ without a repeated header) still extract correctly.
    last_mapping: dict[str, int] = {}

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                for table in (page.extract_tables() or []):
                    if not table or len(table) < 2:
                        continue

                    mapping: dict[str, int] = {}
                    header_idx = -1
                    for row_idx, row in enumerate(table[:5]):
                        m = _map_headers(row)
                        if "date" in m and ("details" in m or "paid_out" in m or "paid_in" in m):
                            mapping = m
                            header_idx = row_idx
                            break

                    if header_idx == -1:
                        # No header — try carry-forward mapping
                        if last_mapping and _is_date_like(_cell(table[0], last_mapping.get("date"))):
                            mapping = last_mapping
                            data_rows = table
                        else:
                            continue
                    else:
                        last_mapping = mapping
                        data_rows = table[header_idx + 1:]

                    found_structure = True
                    for row in data_rows:
                        date = _cell(row, mapping.get("date"))
                        if not _is_date_like(date):
                            continue

                        paid_out_raw = _cell(row, mapping.get("paid_out"))
                        paid_in_raw = _cell(row, mapping.get("paid_in"))
                        details = _cell(row, mapping.get("details"))
                        balance_raw = _cell(row, mapping.get("balance"))
                        payment_type = _cell(row, mapping.get("payment_type"))

                        if not details and not _is_amount_like(paid_out_raw) and not _is_amount_like(paid_in_raw):
                            continue

                        transactions.append(RawTransaction(
                            date=date,
                            payment_type=payment_type,
                            details=details,
                            paid_out=paid_out_raw if _is_amount_like(paid_out_raw) else None,
                            paid_in=paid_in_raw if _is_amount_like(paid_in_raw) else None,
                            balance=balance_raw if _is_amount_like(balance_raw) else None,
                        ))

    except Exception:
        return None

    if not found_structure:
        return None

    has_amounts = any(t.paid_out is not None or t.paid_in is not None for t in transactions)
    return transactions if has_amounts else None


# ─── Text Extraction + OCR Fallback ──────────────────────────────────────────

def extract_text(pdf_path: str) -> str:
    """Extract text via pdfplumber. Falls back to Tesseract OCR for near-blank pages.

    PyMuPDF is used only for page rendering on scanned pages — it doesn't require
    poppler and is already in the dependency stack.
    """
    pages: list[str] = []
    fitz_doc = None  # lazy-open only if OCR is needed

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if len(text.strip()) < 50:
                if fitz_doc is None:
                    fitz_doc = fitz.open(pdf_path)
                pix = fitz_doc[page_num].get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)
            pages.append(text)

    if fitz_doc is not None:
        fitz_doc.close()

    return "\n".join(pages)


# ─── LLM Extraction ───────────────────────────────────────────────────────────

client = AsyncGroq()

_SYSTEM_PROMPT = (
    "Extract all transactions from the bank statement text below. "
    "Map debits/withdrawals/DR to paid_out; credits/deposits/CR to paid_in. "
    "Amounts with CR suffix → paid_in only. Amounts with DR suffix → paid_out only. Store the number only, no suffix.\n"
    "If a chronological transaction list AND grouped summary sections (e.g. 'Deposits', 'Withdrawals', 'Checks Paid') "
    "both exist, extract ONLY from the chronological list — never from the summary sections.\n"
    "A line with no date and no amounts is a continuation of the previous transaction description — "
    "append it to that transaction's details field, do NOT create a new row.\n"
    "Dates as yyyy-mm-dd. Store amounts as strings (e.g. \"12.50\"). Null for missing fields.\n"
    'Return JSON: {"transactions":[{"date":"...","payment_type":"...","details":"...",'
    '"paid_out":"...","paid_in":"...","balance":"..."}]}'
)


async def parse_statement_llm(raw_text: str) -> list[RawTransaction]:
    """Send text to Groq and return a list of RawTransactions."""
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
    raw_list = data.get("transactions", [])

    result: list[RawTransaction] = []
    for item in raw_list:
        try:
            result.append(RawTransaction(**{
                k: str(v) if v is not None else None
                for k, v in item.items()
                if k in RawTransaction.model_fields
            }))
        except Exception:
            continue
    return result


# ─── Post-processing ──────────────────────────────────────────────────────────

def drop_empty_rows(transactions: list[Transaction]) -> list[Transaction]:
    """Drop rows where both paid_in and paid_out are None.

    These are summary lines, totals, or header rows the LLM mistakenly included.
    """
    return [t for t in transactions if t.paid_in is not None or t.paid_out is not None]


# ─── CSV Output ───────────────────────────────────────────────────────────────

_HEADERS = ["date", "payment_type", "details", "paid_out", "paid_in", "balance"]


def write_csv(transactions: list[Transaction], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_HEADERS)
        for t in transactions:
            writer.writerow([
                t.date or "",
                t.payment_type or "",
                t.details or "",
                "" if t.paid_out is None else t.paid_out,
                "" if t.paid_in is None else t.paid_in,
                "" if t.balance is None else t.balance,
            ])


# ─── Main Pipeline ────────────────────────────────────────────────────────────

async def process_pdf(pdf_path: Optional[str]) -> tuple[Optional[str], str]:
    if pdf_path is None:
        return None, "Please upload a PDF file."

    run_id = uuid.uuid4().hex
    tmp_pdf = f"/tmp/stmt_{run_id}.pdf"
    tmp_csv = f"/tmp/out_{run_id}.csv"

    try:
        # Stage 1 — Ingest to /tmp
        with open(pdf_path, "rb") as src, open(tmp_pdf, "wb") as dst:
            dst.write(src.read())

        # Stage 2 — Try deterministic table parser (pdfplumber column-aware)
        raw_transactions = try_extract_tables(tmp_pdf)
        method = "table"

        if raw_transactions is None:
            # Stage 3 — LLM path: extract text, redact PII, call Groq
            raw_text = extract_text(tmp_pdf)
            clean_text = redact_pii(raw_text)
            raw_transactions = await parse_statement_llm(clean_text)
            method = "llm"

        # Normalise amounts deterministically
        transactions = [normalize_raw(r) for r in raw_transactions]

        # Drop phantom rows (summary/total lines with no amounts)
        transactions = drop_empty_rows(transactions)

        # Stage 4 — Write CSV
        write_csv(transactions, tmp_csv)

        count = len(transactions)
        return tmp_csv, f"Done ({method}) — {count} transaction{'s' if count != 1 else ''} extracted."

    except Exception as exc:
        return None, f"Error: {exc}"

    finally:
        if os.path.exists(tmp_pdf):
            os.remove(tmp_pdf)
