"""
pipeline.py - Main pipeline orchestrator for PDF-to-CSV extraction.
Handles routing, normalisation, and CSV writing.
"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import re
import time
import uuid
import zipfile
from typing import Optional

from extraction import extract_text, redact_pii, try_extract_tables
from llm import parse_statement_llm, parse_table_cells
from models import RawTransaction, Transaction, drop_empty_rows, infer_payment_type, normalize_raw


# ─── Fast deterministic path (no LLM) ───────────────────────────────────────
# Checks if table extraction is clean enough to skip LLM

_DATE_LIKE = re.compile(
    r"""\b(
        \d{1,2}[/\-.\s]\d{1,2}[/\-.\s]\d{2,4}   # DD/MM/YYYY or variants
        |\d{4}[/\-.]\d{2}[/\-.\d{2}]             # YYYY-MM-DD
        |\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}      # 01 Jan 2024
        |[A-Za-z]{3,9}\s+\d{1,2}[,\s]+\d{2,4}   # Jan 01, 2024
    )\b""",
    re.VERBOSE,
)
_AMOUNT_LIKE = re.compile(r"[\d,]+\.\d{2}")


def _cells_are_clean(cells: list[list[str]]) -> bool:
    """Check if >=80% of rows have a date and at least one amount column."""
    if not cells:
        return False
    passing = sum(
        1 for row in cells
        if len(row) >= 6
        and _DATE_LIKE.search(row[0] or "")
        and any(_AMOUNT_LIKE.search(row[i] or "") for i in (3, 4, 5))
    )
    return passing / len(cells) >= 0.8


# ─── CSV Output ───────────────────────────────────────────────────────────────

_HEADERS = ["date", "payment_type", "details", "paid_out", "paid_in", "balance"]


def write_csv(transactions: list[Transaction], out_path: str) -> None:
    """Write normalised transactions to a CSV file."""
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


# ─── Confidence Scoring ─────────────────────────────────────────────────────

_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
_DETAIL_PAYMENT_TYPE_RE = re.compile(
    r'^\s*(card payment|transfer|direct debit|standing order|fee|interest|salary|cash withdrawal)\s*(?:[-:–]\s*|\s+)',
    re.I,
)


def _extract_payment_type_from_details(details: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Extract a payment type only when it is explicitly prefixed in details."""
    if not details:
        return None, details

    match = _DETAIL_PAYMENT_TYPE_RE.match(details)
    if not match:
        return None, details

    payment_type = match.group(1).strip().lower()
    cleaned_details = details[match.end():].lstrip()
    return payment_type, cleaned_details or details


def _confidence_score(transactions: list[Transaction], method: str) -> tuple[float, str]:
    """
    Compute a confidence score for the extraction.
    +0.4: table parser succeeded (no LLM)
    +0.2: >=90% dates normalised
    +0.2: >=90% rows have amount
    +0.2: balance reconciles for >80% of rows
    """
    if not transactions:
        return 0.0, "⚠ No data"

    score = 0.0

    # Signal 1 - extraction method
    if method == "table":
        score += 0.4

    # Signal 2 - date quality
    clean_dates = sum(1 for t in transactions if t.date and _DATE_RE.match(t.date))
    if clean_dates / len(transactions) >= 0.9:
        score += 0.2

    # Signal 3 - amount quality
    with_amount = sum(
        1 for t in transactions if t.paid_out is not None or t.paid_in is not None
    )
    if with_amount / len(transactions) >= 0.9:
        score += 0.2

    # Signal 4 - balance reconciliation
    if len(transactions) >= 2:
        pairs = list(zip(transactions, transactions[1:]))
        reconciled = 0
        checkable = 0
        for prev, curr in pairs:
            if prev.balance is not None and curr.balance is not None:
                checkable += 1
                paid_out = prev.paid_out or 0.0
                paid_in = prev.paid_in or 0.0
                expected = prev.balance - paid_out + paid_in
                if abs(expected - curr.balance) <= 0.01:
                    reconciled += 1
        if checkable > 0 and reconciled / checkable >= 0.8:
            score += 0.2
        elif checkable == 0:
            # No balance data - don't penalise, award half
            score += 0.1

    pct = int(score * 100)
    if score >= 0.8:
        label = f"Confidence: {pct}% ✓"
    elif score >= 0.6:
        label = f"Confidence: {pct}% - minor issues, spot-check advised"
    else:
        label = f"Confidence: {pct}% ⚠ Manual review recommended"

    return score, label


# ─── Single-file processor ────────────────────────────────────────────────────

async def _process_single(pdf_path: str, is_scanned: bool = False) -> tuple[list[Transaction], str]:
    """Extract and normalise transactions from one PDF.

    If is_scanned=True, skip table parser and go straight to OCR+LLM.
    Otherwise, try table parser first, fall back to OCR+LLM if needed.

    Returns (transactions, one-line status message).
    """
    run_id = uuid.uuid4().hex[:8]
    tmp_pdf = f"/tmp/stmt_{run_id}.pdf"
    file_name = os.path.basename(pdf_path)

    try:
        with open(pdf_path, "rb") as src, open(tmp_pdf, "wb") as dst:
            dst.write(src.read())

        raw_transactions = None
        method = "table"

        if is_scanned:
            # User indicates scanned: force OCR on every page, bypassing the
            # document-level text gate in extract_text().
            # asyncio.to_thread keeps the event loop free during blocking OCR.
            raw_text = await asyncio.wait_for(
                asyncio.to_thread(extract_text, tmp_pdf, True), timeout=120.0
            )
            clean_text = redact_pii(raw_text)
            raw_transactions = await parse_statement_llm(clean_text)
            method = "ocr+llm"
        else:
            # Try deterministic table extraction first
            cells = await asyncio.wait_for(
                asyncio.to_thread(try_extract_tables, tmp_pdf), timeout=120.0
            )

            if cells is not None:
                if _cells_are_clean(cells):
                    # Fast path: cells are well-structured - skip LLM entirely
                    for cell_row in cells:
                        cell_row[2] = redact_pii(cell_row[2])  # redact details
                    raw_transactions = [
                        RawTransaction(
                            date=r[0], payment_type=r[1], details=r[2],
                            paid_out=r[3], paid_in=r[4], balance=r[5],
                        )
                        for r in cells
                    ]
                    method = "table"
                else:
                    # Cells are messy - let LLM normalise them
                    raw_transactions = await parse_table_cells(cells)
                    method = "table"
            else:
                # No table: fall back to OCR + full LLM extraction
                raw_text = await asyncio.wait_for(
                    asyncio.to_thread(extract_text, tmp_pdf), timeout=120.0
                )
                clean_text = redact_pii(raw_text)
                raw_transactions = await parse_statement_llm(clean_text)
                method = "llm"

        if raw_transactions is None:
            raw_transactions = []


        # --- Document-level date format detection (MM/DD vs DD/MM) ---
        def detect_dayfirst(dates):
            for d in dates:
                if not d or not isinstance(d, str):
                    continue
                m = re.match(r'^(\d{1,2})[/-](\d{1,2})[/-]', d.strip())
                if m:
                    first, second = int(m.group(1)), int(m.group(2))
                    if first > 12:
                        return True
                    elif second > 12:
                        return False
            return True  # default: DD/MM

        all_dates = [r.date for r in raw_transactions if r.date]
        dayfirst = detect_dayfirst(all_dates)

        transactions = [normalize_raw(r, dayfirst=dayfirst) for r in raw_transactions]
        transactions = drop_empty_rows(transactions)
        if method in ("llm", "ocr+llm"):
            # Source has no explicit payment_type column on these routes.
            # Keep payment_type blank instead of inferred labels.
            # But preserve any type text by folding it back into details.
            for t in transactions:
                pt = (t.payment_type or "").strip()
                det = (t.details or "").strip()
                if pt:
                    if det:
                        if not det.lower().startswith(pt.lower()):
                            t.details = f"{pt} - {det}"
                    else:
                        t.details = pt
                t.payment_type = None

        for t in transactions:
            if t.payment_type:
                # Stage 1: normalise raw codes/full text -> canonical label
                # e.g. "DD" -> "direct debit", "Direct Debit" -> "direct debit"
                normalized = infer_payment_type(t.payment_type)
                if normalized:
                    t.payment_type = normalized
                continue

            # Narrow heuristic: only fill payment_type when the details start
            # with one of the explicit labels below. No broader inference.
            extracted_type, cleaned_details = _extract_payment_type_from_details(t.details)
            if extracted_type:
                t.payment_type = extracted_type
                t.details = cleaned_details

        n = len(transactions)
        _, conf_label = _confidence_score(transactions, method)
        return transactions, f"{n} transaction{'s' if n != 1 else ''} ({method})  |  {conf_label}"

    except Exception as exc:
        _logger.warning("Processing failed for %s via %s: %s", file_name, method, exc)
        return [], f"Error: {exc}"

    finally:
        if os.path.exists(tmp_pdf):
            os.remove(tmp_pdf)


# ─── Logging ─────────────────────────────────────────────────────────────────

_logger = logging.getLogger("csvpdf")


def _log(data: dict) -> None:
    """Emit a single-line JSON structured log record."""
    _logger.info(json.dumps(data))


# ─── Preview builder ─────────────────────────────────────────────────────────

def _build_preview(transactions: list[Transaction], max_rows: int = 20) -> list[list[str]]:
    """Return the first *max_rows* transactions as a list-of-lists for gr.Dataframe."""
    rows = []
    for t in transactions[:max_rows]:
        rows.append([
            t.date or "",
            t.payment_type or "",
            t.details or "",
            "" if t.paid_out is None else str(t.paid_out),
            "" if t.paid_in is None else str(t.paid_in),
            "" if t.balance is None else str(t.balance),
        ])
    return rows


# ─── Multi-file entry point ───────────────────────────────────────────────────

async def process_pdfs(
    pdf_paths: Optional[list[str]],
    combine: bool,
    is_scanned: bool = False,
):
    """Async generator - yields (csv_paths, status, zip_path_or_None, preview_rows)
    after each file so the Gradio UI updates in real time.
    Final yield carries complete results including ZIP path (if 2+ files).
    """
    if not pdf_paths:
        yield [], "Please upload at least one PDF.", None, []
        return

    n = len(pdf_paths)
    run_dir = f"/tmp/csvpdf_{uuid.uuid4().hex[:8]}"
    os.makedirs(run_dir, exist_ok=True)

    all_transactions: list[Transaction] = []
    status_lines: list[str] = []
    csv_paths: list[str] = []

    for i, pdf_path in enumerate(pdf_paths):
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        # Progress ping before the slow extraction step
        yield csv_paths, f"[{i + 1}/{n}] Processing {stem}.pdf…", None, _build_preview(all_transactions)

        try:
            t0 = time.time()
            transactions, status = await _process_single(pdf_path, is_scanned=is_scanned)
            elapsed = round(time.time() - t0, 1)

            status_lines.append(f"[{i + 1}/{n}] {stem}.pdf - {status}")
            all_transactions.extend(transactions)
            _log({"file": f"{stem}.pdf", "rows": len(transactions), "elapsed_s": elapsed})

            if not combine:
                out = os.path.join(run_dir, f"{stem}.csv")
                write_csv(transactions, out)
                csv_paths.append(out)

        except asyncio.CancelledError:
            raise  # don't swallow cancellation
        except Exception as exc:
            status_lines.append(f"[{i + 1}/{n}] {stem}.pdf - ERROR: {exc}")
            _log({"file": f"{stem}.pdf", "error": str(exc), "stage": "process_pdfs"})

        # Yield updated state after each file completes (or errors)
        yield csv_paths, "\n".join(status_lines), None, _build_preview(all_transactions)

    if combine:
        name = (
            os.path.splitext(os.path.basename(pdf_paths[0]))[0] + ".csv"
            if len(pdf_paths) == 1
            else "combined.csv"
        )
        out = os.path.join(run_dir, name)
        write_csv(all_transactions, out)
        csv_paths.append(out)

    zip_path: Optional[str] = None
    if len(csv_paths) > 1:
        zip_path = os.path.join(run_dir, "statements.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in csv_paths:
                zf.write(p, os.path.basename(p))

    yield csv_paths, "\n".join(status_lines), zip_path, _build_preview(all_transactions)
