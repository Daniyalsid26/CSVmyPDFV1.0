"""extraction.py — PDF text extraction, OCR fallback, PII redaction, and table parser."""
from __future__ import annotations

import io
import re
from typing import Optional

import fitz  # PyMuPDF — used only for OCR page rendering
import pdfplumber
import pytesseract
from PIL import Image

from models import RawTransaction


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


# ─── Text Extraction + OCR Fallback ──────────────────────────────────────────

_PAGE_SEP = "\n\n--- PAGE {n} ---\n\n"


def extract_text(pdf_path: str) -> str:
    """Extract text via pdfplumber with page-separator markers.

    Routing is document-level: if total pdfplumber text is below 200 chars the
    whole document is re-extracted via OCR, preventing scanned pages with minor
    embedded metadata from bypassing the OCR path.
    """
    per_page: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            per_page.append(page.extract_text() or "")

    if len("".join(per_page).strip()) < 200:
        # Whole document is scanned — re-extract every page via OCR
        fitz_doc = fitz.open(pdf_path)
        try:
            per_page = []
            for i in range(len(fitz_doc)):
                pix = fitz_doc[i].get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                per_page.append(pytesseract.image_to_string(img))
        finally:
            fitz_doc.close()

    return "".join(
        (_PAGE_SEP.format(n=i + 1) if i > 0 else "") + text
        for i, text in enumerate(per_page)
    )


# ─── Deterministic Table Parser ───────────────────────────────────────────────
# Stage 1: pdfplumber's extract_tables() reads column cells by index,
# preserving paid_out/paid_in/balance positions exactly — no LLM guessing.

_HEADER_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bdate\b", re.I), "date"),
    (re.compile(
        r"\b(payment[\s_-]?type|transaction[\s_-]?type|tran[\s_-]?type|txn[\s_-]?type)\b", re.I,
    ), "payment_type"),
    (re.compile(r"\b(type|code|category)\b", re.I), "payment_type"),
    (re.compile(
        r"\b(description|details?|particulars|narrative|reference|memo|payee|remarks|narration)\b", re.I,
    ), "details"),
    (re.compile(
        r"\b(debit|dr|withdrawals?|paid[\s_-]?out|money[\s_-]?out|charges?)\b", re.I,
    ), "paid_out"),
    (re.compile(
        r"\b(credit|cr|deposits?|paid[\s_-]?in|money[\s_-]?in|receipts?|income)\b", re.I,
    ), "paid_in"),
    (re.compile(r"\b(balance|bal)\b", re.I), "balance"),
]


def _match_header(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    h = re.sub(r"\s+", " ", raw.lower().strip())
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

    Returns None when no recognisable table is found, signalling the caller to
    fall through to the LLM path.
    """
    transactions: list[RawTransaction] = []
    found_structure = False
    last_mapping: dict[str, int] = {}  # carry forward across pages

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
