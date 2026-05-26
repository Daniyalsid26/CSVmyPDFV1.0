"""models.py — Pydantic models, amount normalisation, and transaction post-processing."""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel


# ─── Data Models ──────────────────────────────────────────────────────────────

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


# ─── Payment Type Inference ───────────────────────────────────────────────────

_TYPE_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(direct[\s_-]?debit|d\.?d\.?)\b", re.I), "direct debit"),
    (re.compile(r"\b(standing[\s_-]?order|s\.?o\.?)\b", re.I), "standing order"),
    (re.compile(r"\b(bacs|chaps|faster[\s_-]?payment|f\.?p\.?s\.?|transfer|trf)\b", re.I), "transfer"),
    (re.compile(r"\b(card|contactless|visa|mastercard|maestro|pos)\b", re.I), "card payment"),
    (re.compile(r"\b(atm|cash[\s_-]?withdrawal|cashpoint|withdrawal)\b", re.I), "cash withdrawal"),
    (re.compile(r"\b(salary|payroll|wages)\b", re.I), "salary"),
    (re.compile(r"\b(interest)\b", re.I), "interest"),
    (re.compile(r"\b(fee|charges?)\b", re.I), "fee"),
]


def infer_payment_type(details: Optional[str]) -> Optional[str]:
    """Infer a canonical payment_type from a transaction description."""
    if not details:
        return None
    for pattern, label in _TYPE_RULES:
        if pattern.search(details):
            return label
    return None


# ─── Filtering ────────────────────────────────────────────────────────────────

def drop_empty_rows(transactions: list[Transaction]) -> list[Transaction]:
    """Drop rows where both paid_in and paid_out are None (summary/total lines)."""
    return [t for t in transactions if t.paid_in is not None or t.paid_out is not None]
