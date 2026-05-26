"""extraction.py — PDF text extraction, OCR fallback, PII redaction, and table parser."""
from __future__ import annotations

import io
import re
from typing import Optional

import fitz  # PyMuPDF — used only for OCR page rendering
import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance

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
        # Whole document is scanned — re-extract every page via OCR with preprocessing
        fitz_doc = fitz.open(pdf_path)
        try:
            per_page = []
            for i in range(len(fitz_doc)):
                pix = fitz_doc[i].get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                # Enhance contrast and sharpness for better OCR
                contrast_enhancer = ImageEnhance.Contrast(img)
                img = contrast_enhancer.enhance(1.5)
                sharpness_enhancer = ImageEnhance.Sharpness(img)
                img = sharpness_enhancer.enhance(2.0)
                # OCR with PSM 6 (single uniform block) for structured statements
                per_page.append(pytesseract.image_to_string(img, config='--psm 6'))
        finally:
            fitz_doc.close()

    return "".join(
        (_PAGE_SEP.format(n=i + 1) if i > 0 else "") + text
        for i, text in enumerate(per_page)
    )


# ─── Coordinate-Based OCR Row Extractor (for scanned PDFs) ───────────────────

def ocr_extract_rows(pdf_path: str) -> Optional[list[list[str]]]:
    """Extract rows from a scanned PDF using coordinate-aware OCR.

    Uses pytesseract.image_to_data (no pandas required) to get per-word
    positions, then groups words into rows by y-position and assigns them
    to columns by x-position percentage of page width.

    Returns [[date, code, details, paid_out, paid_in, balance], ...] or
    None if the extraction looks empty / unreliable.
    """
    rows: list[list[str]] = []

    try:
        fitz_doc = fitz.open(pdf_path)
        for page_idx in range(len(fitz_doc)):
            pix = fitz_doc[page_idx].get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            # Enhance for better OCR
            img = ImageEnhance.Contrast(img).enhance(1.5)
            img = ImageEnhance.Sharpness(img).enhance(2.0)

            page_w = img.width

            # Get per-word data as dict — no pandas needed
            data = pytesseract.image_to_data(
                img,
                config="--psm 6",
                output_type=pytesseract.Output.DICT,
            )

            # Build list of (top, left, text) for confident, non-empty words
            words: list[tuple[int, int, str]] = []
            for i, text in enumerate(data["text"]):
                text = text.strip()
                if not text:
                    continue
                conf = int(data["conf"][i])
                if conf < 30:
                    continue
                words.append((data["top"][i], data["left"][i], text))

            if not words:
                continue

            # Group words into horizontal rows (words within 10px vertical distance)
            words.sort(key=lambda w: w[0])
            row_groups: list[list[tuple[int, int, str]]] = []
            current_group: list[tuple[int, int, str]] = [words[0]]
            for word in words[1:]:
                if abs(word[0] - current_group[0][0]) <= 10:
                    current_group.append(word)
                else:
                    row_groups.append(current_group)
                    current_group = [word]
            row_groups.append(current_group)

            # Assign words to columns by x-position percentage of page width
            # Bands: date <15% | code 15-22% | details 22-60% |
            #        paid_out 60-75% | paid_in 75-88% | balance 88%+
            for group in row_groups:
                group.sort(key=lambda w: w[1])  # sort left-to-right
                cols: dict[str, list[str]] = {
                    "date": [], "code": [], "details": [],
                    "paid_out": [], "paid_in": [], "balance": [],
                }
                for top, left, text in group:
                    pct = (left / page_w) * 100
                    if pct < 15:
                        cols["date"].append(text)
                    elif pct < 22:
                        cols["code"].append(text)
                    elif pct < 60:
                        cols["details"].append(text)
                    elif pct < 75:
                        cols["paid_out"].append(text)
                    elif pct < 88:
                        cols["paid_in"].append(text)
                    else:
                        cols["balance"].append(text)

                rows.append([
                    " ".join(cols["date"]),
                    " ".join(cols["code"]),
                    " ".join(cols["details"]),
                    " ".join(cols["paid_out"]),
                    " ".join(cols["paid_in"]),
                    " ".join(cols["balance"]),
                ])

        fitz_doc.close()
    except Exception:
        return None

    if not rows:
        return None

    # Merge continuation lines: rows with no date AND no amounts → append
    # details text to the previous row
    merged: list[list[str]] = []
    for row in rows:
        date, code, details, paid_out, paid_in, balance = row
        is_continuation = (
            not date.strip()
            and not _is_amount_like(paid_out)
            and not _is_amount_like(paid_in)
            and details.strip()
        )
        if is_continuation and merged:
            merged[-1][2] = (merged[-1][2] + " " + details).strip()
        else:
            merged.append(row)

    # Keep only rows that look like real transactions (have a date-like value)
    transaction_rows = [r for r in merged if _is_date_like(r[0])]

    # Quality gate: if fewer than 3 date rows found, or >50% of rows have
    # empty paid_out AND paid_in, this extraction is unreliable — return None
    # so the caller can fall back to plain OCR+LLM.
    if len(transaction_rows) < 3:
        return None
    empty_amounts = sum(
        1 for r in transaction_rows
        if not _is_amount_like(r[3]) and not _is_amount_like(r[4])
    )
    if empty_amounts / len(transaction_rows) > 0.5:
        return None

    return transaction_rows


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


def try_extract_tables(pdf_path: str) -> Optional[list[list[str]]]:
    """Attempt deterministic table structure detection.

    Returns raw cell data (not parsed transactions) when a table is found.
    Format: [[date, type, details, paid_out, paid_in, balance], ...]
    Returns None when no recognisable table is found, signalling the caller to
    fall through to the OCR → LLM path.
    """
    cells: list[list[str]] = []
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

                        paid_out_raw = _cell(row, mapping.get("paid_out")) or ""
                        paid_in_raw = _cell(row, mapping.get("paid_in")) or ""
                        details = _cell(row, mapping.get("details")) or ""
                        balance_raw = _cell(row, mapping.get("balance")) or ""
                        payment_type = _cell(row, mapping.get("payment_type")) or ""

                        if not details and not _is_amount_like(paid_out_raw) and not _is_amount_like(paid_in_raw):
                            continue

                        cells.append([date or "", payment_type, details, paid_out_raw, paid_in_raw, balance_raw])

    except Exception:
        return None

    if not found_structure:
        return None

    has_amounts = any(row[3] or row[4] for row in cells if len(row) >= 5)
    return cells if has_amounts else None
