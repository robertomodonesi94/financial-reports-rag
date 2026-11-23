"""Query-side improvements for retrieval over financial reports.

- expand_glossary: add the line-item names that filings actually use ("capital expenditure" is printed as
  "purchases of property, plant and equipment" or "capital spending"). Deterministic, no model needed.
- rewrite_with_llm: let the language model rewrite the question as a search query in filing vocabulary.
- guess_docs: find the company and the fiscal years named in the question and keep only those reports,
  so a question about 3M in 2018 is not answered from Amcor's or 3M's 2022 report.
"""
from __future__ import annotations

import re

from .llm import LLM

# (pattern in the question, terms used in SEC filings)
GLOSSARY: list[tuple[str, str]] = [
    (r"\bcap(?:ital)?[\s-]?ex\b|capital expenditures?", "purchases of property, plant and equipment capital spending additions to property and equipment"),
    (r"\bcogs\b|cost of goods sold", "cost of goods sold cost of sales cost of revenue"),
    (r"\bebitda\b", "operating income depreciation and amortization"),
    (r"\bd&a\b|depreciation and amortization", "depreciation and amortization"),
    (r"\bfree cash ?flow\b|\bfcf\b", "net cash provided by operating activities purchases of property and equipment"),
    (r"\bquick ratio\b", "cash and cash equivalents short-term investments accounts receivable total current liabilities"),
    (r"\bcurrent ratio\b|\bliquidity\b|\bworking capital\b", "total current assets total current liabilities"),
    (r"\bgross margins?\b|\bgross profit\b", "gross profit gross margin net sales cost of sales"),
    (r"\boperating margins?\b|\boperating income\b", "operating income operating profit net revenues"),
    (r"\beffective tax rate\b", "effective tax rate provision for income taxes income before income taxes"),
    (r"\bdays payable outstanding\b|\bdpo\b", "accounts payable cost of sales"),
    (r"\binventory turnover\b|\bdays inventory\b", "inventories cost of sales"),
    (r"\bdividends?\b", "dividends declared paid per share"),
    (r"\bcash (?:and|&) cash equivalents\b", "cash and cash equivalents"),
    (r"\brevenues?\b|\bnet sales\b|\btop ?line\b", "net revenues net sales total revenues"),
    (r"\bshare repurchases?\b|\bbuy ?backs?\b", "repurchases of common stock treasury stock"),
    (r"\bsegments?\b", "reportable segments segment information"),
    (r"\brestructuring\b", "restructuring charges restructuring liability"),
    (r"\bcash flow statement\b|\bstatement of cash flows?\b", "consolidated statement of cash flows investing activities operating activities"),
    (r"\bincome statement\b|\bp&l\b", "consolidated statement of income"),
    (r"\bbalance sheet\b", "consolidated balance sheet total assets total liabilities"),
    (r"\bbook value\b|\bliquidat\w*\b", "total shareholders equity book value per share common shares outstanding"),
    (r"\bspin(?:ning)?[\s-]?off\b|\bseparation\b|\bdivest\w*", "separation spin-off discontinued operations divestiture"),
]
_GLOSSARY_RE = [(re.compile(p, re.I), terms) for p, terms in GLOSSARY]

REWRITE_PROMPT = (
    "Rewrite the question below as a short search query (at most 25 words) for a company's SEC filing (10-K, 10-Q, 8-K). "
    "Use the exact line-item names that filings print, e.g. 'capital expenditure' -> 'purchases of property, plant and equipment', "
    "'COGS' -> 'cost of sales'. Keep the company name and the fiscal years. Reply with the query only.\n\nQuestion: {question}"
)

# question spellings -> company name as stored in the document metadata
ALIASES = {
    "amex": "American Express", "jpm": "JPMorgan", "jp morgan": "JPMorgan", "jnj": "Johnson & Johnson",
    "j and j": "Johnson & Johnson", "coca cola": "Coca-Cola", "coke": "Coca-Cola", "mgm": "MGM Resorts",
    "aes": "AES Corporation", "cvs": "CVS Health", "ulta": "Ulta Beauty", "kraft": "Kraft Heinz",
    "lockheed": "Lockheed Martin", "american water": "American Water Works", "activision": "Activision Blizzard",
    "general mills": "General Mills", "foot locker": "Foot Locker", "best buy": "Best Buy",
}


def expand_glossary(question: str) -> str:
    """Filing vocabulary for the financial terms found in the question ('' when none is found)."""
    terms = [t for rx, t in _GLOSSARY_RE if rx.search(question)]
    return " ".join(dict.fromkeys(terms))


def rewrite_with_llm(question: str, llm: LLM, max_new_tokens: int = 60) -> str:
    """One short LLM call; returns '' if the model replies with nothing usable."""
    try:
        out = llm.chat([{"role": "user", "content": REWRITE_PROMPT.format(question=question)}], max_new_tokens=max_new_tokens)
    except Exception:
        return ""
    line = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
    line = re.sub(r"^(search )?query\s*:\s*", "", line, flags=re.I).strip(" \"'")
    return line[:300]


def _norm(text: str) -> str:
    text = text.lower().replace("&", " and ")
    return " " + re.sub(r"[^a-z0-9]+", " ", text).strip() + " "


def years_in(question: str) -> set[int]:
    """Fiscal years named in the question: FY2022, FY22, FY 2022, 2022, Q2'2023, Q2 of FY2024..."""
    years = {int(y) for y in re.findall(r"\b(20\d\d)\b", question)}
    years |= {2000 + int(y) for y in re.findall(r"\bFY\s?'?(\d{2})\b", question, re.I)}
    years |= {int(y) for y in re.findall(r"(?:FY|Q[1-4])\s?'?\s?(20\d\d)", question, re.I)}
    return {y for y in years if 1990 <= y <= 2035}


def guess_docs(question: str, docs: dict[str, dict]) -> list[str] | None:
    """Reports of the company (and fiscal years) named in the question; None when no company is recognised."""
    q = _norm(question)
    names: dict[str, str] = {}
    for doc_id, d in docs.items():
        company = d.get("company") or doc_id.split("_")[0]
        names[_norm(company).strip()] = company
        names[_norm(doc_id.split("_")[0]).strip()] = company      # e.g. PEPSICO, BESTBUY
    names.update({_norm(a).strip(): c for a, c in ALIASES.items()})
    hit = {company for name, company in names.items() if name and f" {name} " in q}
    if not hit:
        return None
    wanted = {c.lower() for c in hit}
    candidates = {i: d for i, d in docs.items() if (d.get("company") or i.split("_")[0]).lower() in wanted}
    years = years_in(question)
    if years:
        same_year = [i for i, d in candidates.items() if isinstance(d.get("period"), int) and d["period"] in years]
        if same_year:
            return sorted(same_year)
    return sorted(candidates) or None
