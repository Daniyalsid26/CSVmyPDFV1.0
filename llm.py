"""llm.py — Groq LLM client, system prompt, and transaction extraction."""
from __future__ import annotations

import asyncio
import json

from groq import AsyncGroq

from models import RawTransaction

client = AsyncGroq()
_LLM_SEM = asyncio.Semaphore(3)  # max 3 concurrent Groq requests

_SYSTEM_PROMPT = (
    "You are a bank statement parser. Extract ALL transaction rows from the text below.\n"
    "Return a JSON object: {\"transactions\": [{\"date\": \"2019-01-15\", \"payment_type\": \"card payment\", "
    "\"details\": \"TESCO STORES\", \"paid_out\": \"12.50\", \"paid_in\": null, \"balance\": \"987.50\"}]}\n"
    "CR/DR Rule: amounts with CR suffix → paid_in only; DR suffix → paid_out only. Store number only, no suffix.\n"
    "Year Rule: many statements print only day+month per row; the year appears in the statement header or period line. "
    "Find it and apply it to every transaction date. Never use today's year as a substitute.\n"
    "Balance Rule: extract the running balance on each transaction row into the balance field. "
    "Do NOT include the opening/closing balance summary lines as transaction rows.\n"
    "Duplicate Section Rule: if a chronological list AND grouped summary sections (Deposits, Withdrawals, Checks Paid) "
    "both exist, extract ONLY from the chronological list.\n"
    "Continuation Line Rule: a line with no date and no amounts is part of the previous transaction — "
    "append it to that transaction's details, do NOT create a new row.\n"
    "Payment Type Rule: if no explicit column, infer from description using exactly these labels: "
    "card payment, transfer, direct debit, standing order, cash withdrawal, salary, interest, fee. Null if unclear.\n"
    "Dates as yyyy-mm-dd. Amounts as strings. Null for missing fields."
)

_TABLE_CELL_PROMPT = (
    "You receive 6 columns per row: [date, payment_type, details, paid_out, paid_in, balance]\n"
    "Your task: normalise and return as JSON.\n"
    "Return a JSON object: {\"transactions\": [{\"date\": \"YYYY-MM-DD\", \"payment_type\": \"...\", "
    "\"details\": \"...\", \"paid_out\": \"...\", \"paid_in\": \"...\", \"balance\": \"...\"}]}\n"
    "Rules:\n"
    "- Date: convert to YYYY-MM-DD. If only DD Mmm YY, infer year from document context (ask: what year is this statement from?).\n"
    "- Amounts: strip currency symbols, CRs/DRs, whitespace. Keep numeric only, as string.\n"
    "- Details: keep full text, strip leading/trailing whitespace.\n"
    "- Balance: keep as-is if present, else null.\n"
    "- Payment type: normalise to one of: direct debit, card payment, transfer, standing order, cash withdrawal, salary, interest, fee. If null in input, infer from details or return null.\n"
    "Only return valid JSON. Output Null for any missing or empty field."
)


async def parse_statement_llm(raw_text: str) -> list[RawTransaction]:
    """Send text to Groq and return a list of RawTransactions."""
    if len(raw_text) > 90_000:
        raw_text = raw_text[:90_000]

    async with _LLM_SEM:
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            timeout=120.0,
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


_CLASSIFY_PROMPT = (
    "Classify each transaction description into exactly one payment type.\n"
    "Use only these labels: card payment, transfer, direct debit, standing order, "
    "cash withdrawal, salary, interest, fee, bill payment, subscription, insurance, other\n"
    "Return JSON: {\"results\": {\"0\": \"bill payment\", \"1\": \"card payment\", ...}}\n"
    "Use the integer key from the input. Never add extra keys or explanations."
)


async def classify_payment_types(descriptions: list[str]) -> dict[int, str]:
    """Batch-classify a list of transaction descriptions into payment type labels.

    Sends a single Groq call for all descriptions. Returns a dict mapping
    the original list index (int) to a canonical payment type label (str).
    Returns an empty dict on any failure so callers degrade gracefully.
    """
    if not descriptions:
        return {}

    numbered = "\n".join(f"{i}: {desc}" for i, desc in enumerate(descriptions))

    try:
        async with _LLM_SEM:
            response = await client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": _CLASSIFY_PROMPT},
                    {"role": "user", "content": numbered},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                timeout=120.0,
            )
        data = json.loads(response.choices[0].message.content)
        raw = data.get("results", {})
        return {int(k): str(v) for k, v in raw.items() if str(k).isdigit()}
    except Exception:
        return {}
async def parse_table_cells(cells: list[list[str]]) -> list[RawTransaction]:
    """Parse pre-separated table cells from structured table extraction.

    Cells format: [[date, payment_type, details, paid_out, paid_in, balance], ...]
    Returns a list of RawTransactions with normalised values.
    """
    # Format cells as newline-delimited for the LLM prompt
    cell_text = "\n".join(
        f"[{cell[0]!r}, {cell[1]!r}, {cell[2]!r}, {cell[3]!r}, {cell[4]!r}, {cell[5]!r}]"
        for cell in cells
    )

    async with _LLM_SEM:
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _TABLE_CELL_PROMPT},
                {"role": "user", "content": cell_text},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            timeout=120.0,
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
