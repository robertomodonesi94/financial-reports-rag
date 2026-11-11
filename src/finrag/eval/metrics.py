"""Metrics. Simple, transparent and cheap to compute; they complement (do not replace) reading the answers."""
from __future__ import annotations

import re
from collections import Counter
from statistics import mean

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_NUMBER = re.compile(r"(\()?(-)?(\$)?\s?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(\))?\s?(%)?")


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s.%]", " ", text)
    text = _ARTICLES.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def numbers(text: str) -> list[float]:
    """Numbers in a text: '$1,577.00' -> 1577.0, '(12.5)' -> -12.5, '8.7%' -> 8.7. Bare years (1990-2035) are skipped."""
    out = []
    for m in _NUMBER.finditer(text.replace("−", "-")):
        paren, minus, dollar, integer, decimals, close, pct = m.groups()
        value = float(integer.replace(",", "") + (decimals or ""))
        if not (dollar or decimals or pct or "," in integer) and 1990 <= value <= 2035:
            continue  # FY2022, Q2 2023...
        if minus or (paren and close):  # accounting style: (12.5) is negative
            value = -value
        out.append(value)
    return out


def number_match(pred: str, gold: str, rel_tol: float = 0.01) -> bool | None:
    """True when the answer contains the gold answer's first number, within rel_tol.
    Lenient on scale: x100 (ratio vs percent) and x1000 (million vs billion) also match. None if the gold has no number."""
    golds = numbers(gold)
    if not golds:
        return None
    g = golds[0]
    for p in numbers(pred):
        for scale in (1, 100, 0.01, 1000, 0.001):
            target = g * scale
            if abs(p - target) <= rel_tol * max(abs(target), 1e-9) or (target == 0 and abs(p) < 1e-9):
                return True
    return False


def doc_hit(retrieved_doc_ids: list[str], gold_ids: list[str]) -> bool:
    return any(g in retrieved_doc_ids for g in gold_ids)


def page_hit(retrieved_pages: list[tuple[str, int]], gold_pages: list) -> bool | None:
    """True when a retrieved chunk lies on a page that holds the gold evidence. None if the gold has no pages."""
    if not gold_pages:
        return None
    gold = {(d, int(p)) for d, p in gold_pages}
    return any((d, int(p)) in gold for d, p in retrieved_pages)


def citation_precision(cited_doc_ids: list[str], gold_ids: list[str]) -> float | None:
    """Share of cited documents that are gold documents; None when nothing was cited."""
    if not cited_doc_ids:
        return None
    return sum(c in gold_ids for c in cited_doc_ids) / len(cited_doc_ids)


def _avg(values: list) -> float | None:
    values = [float(v) for v in values if v is not None]
    return round(mean(values), 3) if values else None


def summarize(rows: list[dict]) -> dict:
    """rows: one dict per question with keys type, abstained, f1, number_match, doc_hit, page_hit,
    citation_precision, invalid_citations (the optional ones may be missing or None)."""
    ans = [r for r in rows if r["type"] != "unanswerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]
    return {
        "n_answerable": len(ans),
        "n_unanswerable": len(una),
        "doc_hit": _avg([r.get("doc_hit") for r in ans]),
        "page_hit": _avg([r.get("page_hit") for r in ans]),
        "number_match": _avg([r.get("number_match") for r in ans]),
        "f1": _avg([r["f1"] for r in ans]),
        "citation_precision": _avg([r.get("citation_precision") for r in ans]),
        "false_abstention": _avg([r["abstained"] for r in ans]),
        "abstention_accuracy": _avg([r["abstained"] for r in una]),
        "invalid_citation_rate": _avg([bool(r.get("invalid_citations")) for r in ans]),
    }
