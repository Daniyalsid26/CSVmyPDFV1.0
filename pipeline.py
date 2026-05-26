"""pipeline.py — Main pipeline orchestrator."""
from __future__ import annotations

import csv
import os
import uuid
from typing import Optional

from extraction import extract_text, redact_pii, try_extract_tables
from llm import parse_statement_llm
from models import Transaction, drop_empty_rows, infer_payment_type, normalize_raw


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


# ─── Single-file processor ────────────────────────────────────────────────────

async def _process_single(pdf_path: str) -> tuple[list[Transaction], str]:
    """Extract and normalise transactions from one PDF.

    Returns (transactions, one-line status message).
    """
    run_id = uuid.uuid4().hex[:8]
    tmp_pdf = f"/tmp/stmt_{run_id}.pdf"

    try:
        with open(pdf_path, "rb") as src, open(tmp_pdf, "wb") as dst:
            dst.write(src.read())

        raw_transactions = try_extract_tables(tmp_pdf)
        method = "table"

        if raw_transactions is None:
            raw_text = extract_text(tmp_pdf)
            clean_text = redact_pii(raw_text)
            raw_transactions = await parse_statement_llm(clean_text)
            method = "llm"

        transactions = [normalize_raw(r) for r in raw_transactions]
        transactions = drop_empty_rows(transactions)
        for t in transactions:
            if t.payment_type is None:
                t.payment_type = infer_payment_type(t.details)

        n = len(transactions)
        return transactions, f"{n} transaction{'s' if n != 1 else ''} ({method})"

    except Exception as exc:
        return [], f"Error: {exc}"

    finally:
        if os.path.exists(tmp_pdf):
            os.remove(tmp_pdf)


# ─── Multi-file entry point ───────────────────────────────────────────────────

async def process_pdfs(
    pdf_paths: Optional[list[str]],
    combine: bool,
) -> tuple[list[str], str]:
    """Process one or more PDFs. Called directly by Gradio.

    Returns (list_of_csv_paths, status_text).
    When combine=True all transactions are merged into one CSV named after
    the single file (if one) or "combined.csv" (if many).
    When combine=False each PDF gets its own CSV named "{stem}.csv".
    """
    if not pdf_paths:
        return [], "Please upload at least one PDF."

    run_dir = f"/tmp/csvpdf_{uuid.uuid4().hex[:8]}"
    os.makedirs(run_dir, exist_ok=True)

    all_transactions: list[Transaction] = []
    status_lines: list[str] = []
    csv_paths: list[str] = []

    for pdf_path in pdf_paths:
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        transactions, status = await _process_single(pdf_path)
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

    return csv_paths, "\n".join(status_lines)
