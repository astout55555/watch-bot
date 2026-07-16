"""Bill reference handling.

GovQL identifies bills as ``[type][number]-[congress]`` (e.g. ``hr1181-119``),
following the unitedstates/congress convention. congress.gov uses uppercase
type codes ("HR", "SJRES"); official prose uses dotted forms ("H.R.", "S.J.Res.").
GovQL's ``bills`` table is currently empty and ``votes.relatedBillId`` is null,
so the working join between our bill index and GovQL votes is textual: vote
``question`` strings embed the bill reference in either House clerk style
("H R 1181") or dotted style ("H.R. 1181"). This module converts between all
of these forms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# type -> (spaced clerk style pieces, dotted style)
_STYLE = {
    "hr": ("H R", "H.R."),
    "s": ("S", "S."),
    "hres": ("H RES", "H.Res."),
    "sres": ("S RES", "S.Res."),
    "hjres": ("H J RES", "H.J.Res."),
    "sjres": ("S J RES", "S.J.Res."),
    "hconres": ("H CON RES", "H.Con.Res."),
    "sconres": ("S CON RES", "S.Con.Res."),
}

BILL_TYPES = tuple(_STYLE)

_BILL_ID_RE = re.compile(r"^([a-z]+)(\d+)-(\d+)\Z")


@dataclass(frozen=True)
class BillRef:
    bill_type: str
    number: int
    congress: int

    @property
    def bill_id(self) -> str:
        return f"{self.bill_type}{self.number}-{self.congress}"


def _normalize_type(bill_type: str) -> str:
    normalized = re.sub(r"[.\s]", "", bill_type).lower()
    if normalized not in _STYLE:
        raise ValueError(f"Unknown bill type: {bill_type!r}")
    return normalized


def canonical_bill_id(bill_type: str, number: int, congress: int) -> str:
    """Build a GovQL-style bill_id like ``hr1181-119`` from any common type spelling."""
    if number <= 0:
        raise ValueError(f"Bill number must be positive, got {number}")
    return BillRef(_normalize_type(bill_type), number, congress).bill_id


def parse_bill_id(bill_id: str) -> BillRef:
    """Parse a canonical bill_id like ``hr1181-119`` into its parts."""
    match = _BILL_ID_RE.match(bill_id)
    if not match:
        raise ValueError(f"Malformed bill_id: {bill_id!r}")
    bill_type, number, congress = match.groups()
    if bill_type not in _STYLE:
        raise ValueError(f"Unknown bill type in bill_id: {bill_id!r}")
    return BillRef(bill_type, int(number), int(congress))


def question_reference_variants(ref: BillRef) -> list[str]:
    """Textual forms of the bill reference as they appear in vote question text."""
    spaced, dotted = _STYLE[ref.bill_type]
    return [f"{spaced} {ref.number}", f"{dotted} {ref.number}"]


def display_name(ref: BillRef) -> str:
    """Human-facing citation form, e.g. ``H.R. 1181``."""
    return f"{_STYLE[ref.bill_type][1]} {ref.number}"
