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

        # Infer payment_type from details where not set
        for t in transactions:
            if t.payment_type is None:
                t.payment_type = infer_payment_type(t.details)

        # Stage 4 — Write CSV
        write_csv(transactions, tmp_csv)

        count = len(transactions)
        return tmp_csv, f"Done ({method}) — {count} transaction{'s' if count != 1 else ''} extracted."

    except Exception as exc:
        return None, f"Error: {exc}"

    finally:
        if os.path.exists(tmp_pdf):
            os.remove(tmp_pdf)
