"""llm.py — Groq LLM client, system prompt, and transaction extraction."""
from __future__ import annotations

import json

from groq import AsyncGroq

from models import RawTransaction

client = AsyncGroq()

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
