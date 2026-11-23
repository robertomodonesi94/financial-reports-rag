"""Run a gold dataset through the RAG pipeline or the agent and compare configurations.

scope="shared": search all reports at once (harder, realistic for a company-wide assistant).
scope="per_doc": search only the report(s) the question is about (FinanceBench's "single store" setting): an upper bound,
because in real use nobody hands you the right report.
mode="retrieval" scores only the search (no answer is generated): fast enough to compare many variants on every question.
"""
from __future__ import annotations

from ..pipeline import Pipeline
from ..rag import is_abstention
from .metrics import citation_precision, doc_hit, number_match, page_hit, summarize, token_f1


def _score_common(ex: dict, text: str, abstained: bool) -> dict:
    gold = ex.get("answer", "")
    nm = number_match(text, gold) if gold else None   # None when the gold answer has no number
    if abstained and nm is not None:
        nm = False
    return {"id": ex["id"], "type": ex["type"], "question": ex["question"], "gold": gold, "answer": text, "abstained": abstained,
            "f1": token_f1(text, gold) if gold and not abstained else 0.0, "number_match": nm}


def run_retrieval(ex: dict, rag, scope: str = "shared") -> dict:
    hits, info = rag.retrieve(ex["question"], doc_ids=ex["doc_ids"] if scope == "per_doc" and ex["doc_ids"] else None)
    ph = page_hit([(h.chunk.doc_id, h.chunk.page) for h in hits], ex.get("pages", []))
    flt = info["doc_filter"]
    return {"id": ex["id"], "type": ex["type"], "question": ex["question"],
            "doc_hit": float(doc_hit([h.chunk.doc_id for h in hits], ex["doc_ids"])),
            "page_hit": None if ph is None else float(ph),
            "filter_used": float(info["filter_source"] == "guessed"),
            "filter_correct": None if info["filter_source"] != "guessed" else float(set(ex["doc_ids"]) <= set(flt or [])),
            "search_queries": info["search_queries"], "doc_filter": flt,
            "sources": [f"{h.chunk.doc_id} p{h.chunk.page}" for h in hits]}


def summarize_retrieval(rows: list[dict]) -> dict:
    def avg(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None
    return {"n_answerable": len(rows), "n_unanswerable": 0, "doc_hit": avg("doc_hit"), "page_hit": avg("page_hit"),
            "filter_used": avg("filter_used"), "filter_correct": avg("filter_correct")}


def run_rag(ex: dict, rag, scope: str = "shared") -> dict:
    a = rag.ask(ex["question"], doc_ids=ex["doc_ids"] if scope == "per_doc" and ex["doc_ids"] else None)
    row = _score_common(ex, a.text, a.abstained)
    if ex["type"] != "unanswerable":
        row["doc_hit"] = float(doc_hit([s["doc_id"] for s in a.sources], ex["doc_ids"]))
        ph = page_hit([(s["doc_id"], s["page"]) for s in a.sources], ex.get("pages", []))
        row["page_hit"] = None if ph is None else float(ph)
        row["citation_precision"] = citation_precision(a.cited_doc_ids(), ex["doc_ids"])
    row["invalid_citations"] = a.invalid_citations
    row["sources"] = [f"{s['doc_id']} p{s['page']}" for s in a.sources]
    row["search_queries"], row["doc_filter"] = a.search_queries, a.doc_filter
    return row


def run_agent(ex: dict, agent, scope: str = "shared") -> dict:
    question = ex["question"]
    if scope == "per_doc" and ex["doc_ids"]:
        question += f"\n(The relevant report: {', '.join(ex['doc_ids'])}.)"
    r = agent.run(question)
    row = _score_common(ex, r.answer, (not r.finished) or is_abstention(r.answer))
    if ex["type"] != "unanswerable":
        row["citation_precision"] = citation_precision(r.cited_ids, ex["doc_ids"])
    row["invalid_citations"] = r.unsupported_ids
    row["steps"] = len(r.steps)
    row["invalid_steps"] = sum(1 for s in r.steps if s["type"] == "invalid")
    return row


def evaluate(examples: list[dict], pipeline: Pipeline, mode: str = "rag", scope: str = "shared", **overrides) -> dict:
    if mode == "retrieval":
        rag = pipeline.rag(**overrides)
        rows = [run_retrieval(ex, rag, scope) for ex in examples if ex["type"] != "unanswerable"]
        return {"summary": summarize_retrieval(rows), "rows": rows}
    runner, system = (run_rag, pipeline.rag(**overrides)) if mode == "rag" else (run_agent, pipeline.agent(**overrides))
    rows = [runner(ex, system, scope) for ex in examples]
    return {"summary": summarize(rows), "rows": rows}


def compare(examples: list[dict], pipeline: Pipeline, variants: dict[str, dict], mode: str = "rag") -> dict[str, dict]:
    """variants: name -> overrides; the special key 'scope' selects shared or per_doc search."""
    out = {}
    for name, overrides in variants.items():
        overrides = dict(overrides)
        scope = overrides.pop("scope", "shared")
        out[name] = evaluate(examples, pipeline, mode, scope, **overrides)
    return out


COLUMNS = ["doc_hit", "page_hit", "number_match", "f1", "citation_precision", "false_abstention", "abstention_accuracy", "invalid_citation_rate"]
RETRIEVAL_COLUMNS = ["doc_hit", "page_hit", "filter_used", "filter_correct"]


def to_markdown(results: dict[str, dict]) -> str:
    first_summary = next(iter(results.values()))["summary"]
    cols = RETRIEVAL_COLUMNS if "filter_used" in first_summary else COLUMNS
    head = "| variant | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols)
    lines = [head]
    for name, res in results.items():
        s = res["summary"]
        lines.append(f"| {name} | " + " | ".join("-" if s[c] is None else f"{s[c]:.2f}" for c in cols) + " |")
    first = next(iter(results.values()))["summary"]
    lines.append(f"\n{first['n_answerable']} answerable and {first['n_unanswerable']} unanswerable questions.")
    return "\n".join(lines)
