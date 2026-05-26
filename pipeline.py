"""pipeline.py — Main pipeline orchestrator."""
from __future__ import annotations

import csv
import os
import re
import uuid
import zipfile
from typing import Optional

from extraction import extract_text, redact_pii, try_extract_tables
from llm import classify_payment_types, parse_statement_llm, parse_table_cells
from models import RawTransaction, Transaction, drop_empty_rows, infer_payment_type, normalize_raw


# ─── Fast deterministic path (no LLM) ───────────────────────────────────────

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
    """Return True if >=80% of rows have a date-like col[0] and at least one
    amount-like value in col[3], col[4], or col[5].  When True we can skip
    the LLM and normalise directly in Python."""
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


def _confidence_score(transactions: list[Transaction], method: str) -> tuple[float, str]:
    """Return (score 0.0-1.0, human-readable label).

    Scoring breakdown (all-or-nothing per signal):
      +0.4  table parser succeeded (no LLM needed)
      +0.2  >=90% of dates normalised to YYYY-MM-DD
      +0.2  >=90% of rows have at least one amount (paid_out or paid_in)
      +0.2  balance column reconciles for >80% of consecutive row pairs
    """
    if not transactions:
        return 0.0, "⚠ No data"

    score = 0.0

    # Signal 1 — extraction method
    if method == "table":
        score += 0.4

    # Signal 2 — date quality
    clean_dates = sum(1 for t in transactions if t.date and _DATE_RE.match(t.date))
    if clean_dates / len(transactions) >= 0.9:
        score += 0.2

    # Signal 3 — amount quality
    with_amount = sum(
        1 for t in transactions if t.paid_out is not None or t.paid_in is not None
    )
    if with_amount / len(transactions) >= 0.9:
        score += 0.2

    # Signal 4 — balance reconciliation
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
            # No balance data — don't penalise, award half
            score += 0.1

    pct = int(score * 100)
    if score >= 0.8:
        label = f"Confidence: {pct}% ✓"
    elif score >= 0.6:
        label = f"Confidence: {pct}% — minor issues, spot-check advised"
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

    try:
        with open(pdf_path, "rb") as src, open(tmp_pdf, "wb") as dst:
            dst.write(src.read())

        raw_transactions = None
        method = "table"

        if is_scanned:
            # User indicates scanned: force OCR on every page, bypassing the
            # document-level text gate in extract_text().
            raw_text = extract_text(tmp_pdf, force_ocr=True)
            clean_text = redact_pii(raw_text)
            raw_transactions = await parse_statement_llm(clean_text)
            method = "ocr+llm"
        else:
            # Try deterministic table extraction first
            cells = try_extract_tables(tmp_pdf)

            if cells is not None:
                if _cells_are_clean(cells):
                    # Fast path: cells are well-structured — skip LLM entirely
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
                    # Cells are messy — let LLM normalise them
                    raw_transactions = await parse_table_cells(cells)
                    method = "table"
            else:
                # No table: fall back to OCR + full LLM extraction
                raw_text = extract_text(tmp_pdf)
                clean_text = redact_pii(raw_text)
                raw_transactions = await parse_statement_llm(clean_text)
                method = "llm"

        if raw_transactions is None:
            raw_transactions = []

        transactions = [normalize_raw(r) for r in raw_transactions]
        transactions = drop_empty_rows(transactions)
        for t in transactions:
            # Stage 1: normalise raw codes/full text → canonical label
            # e.g. "DD" → "direct debit", "Direct Debit" → "direct debit"
            normalized = infer_payment_type(t.payment_type)
            if normalized:
                t.payment_type = normalized
            elif not t.payment_type:
                # No payment type column — infer from details text
                t.payment_type = infer_payment_type(t.details)

        # Stage 2: batch LLM fallback for anything still unclassified
        unclassified_indices = [
            i for i, t in enumerate(transactions) if not t.payment_type
        ]
        if unclassified_indices:
            descriptions = [transactions[i].details or "" for i in unclassified_indices]
            llm_labels = await classify_payment_types(descriptions)
            for batch_pos, tx_idx in enumerate(unclassified_indices):
                label = llm_labels.get(batch_pos)
                if label:
                    transactions[tx_idx].payment_type = label

        n = len(transactions)
        _, conf_label = _confidence_score(transactions, method)
        return transactions, f"{n} transaction{'s' if n != 1 else ''} ({method})  |  {conf_label}"

    except Exception as exc:
        return [], f"Error: {exc}"

    finally:
        if os.path.exists(tmp_pdf):
            os.remove(tmp_pdf)


# ─── Multi-file entry point ───────────────────────────────────────────────────

async def process_pdfs(
    pdf_paths: Optional[list[str]],
    combine: bool,
    is_scanned: bool = False,
) -> tuple[list[str], str, Optional[str]]:
    """Process one or more PDFs. Called directly by Gradio.

    Returns (list_of_csv_paths, status_text, zip_path_or_None).
    zip_path is set when there are 2+ CSVs; None for a single file.
    When combine=True all transactions are merged into one CSV named after
    the single file (if one) or "combined.csv" (if many).
    When combine=False each PDF gets its own CSV named "{stem}.csv".
    If is_scanned=True, skip table parsing and use OCR+LLM for all files.
    """
    if not pdf_paths:
        return [], "Please upload at least one PDF.", None

    run_dir = f"/tmp/csvpdf_{uuid.uuid4().hex[:8]}"
    os.makedirs(run_dir, exist_ok=True)

    all_transactions: list[Transaction] = []
    status_lines: list[str] = []
    csv_paths: list[str] = []

    for pdf_path in pdf_paths:
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        transactions, status = await _process_single(pdf_path, is_scanned=is_scanned)
        status_lines.append(f"{stem}.pdf  —  {status}")
        all_transactions.extend(transactions)

        if not combine:
            out = os.path.join(run_dir, f"{stem}.csv")
            write_csv(transactions, out)
            csv_paths.append(out)

    if combine:
        if len(pdf_paths) == 1:
            name = os.path.splitext(os.path.basename(pdf_paths[0]))[0] + ".csv"
        else:
            name = "combined.csv"
        out = os.path.join(run_dir, name)
        write_csv(all_transactions, out)
        csv_paths.append(out)

    # Build ZIP when there are multiple CSVs — created server-side at
    # conversion time so Download All requires no browser round-trip.
    zip_path: Optional[str] = None
    if len(csv_paths) > 1:
        zip_path = os.path.join(run_dir, "statements.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in csv_paths:
                zf.write(p, os.path.basename(p))

    return csv_paths, "\n".join(status_lines), zip_path
