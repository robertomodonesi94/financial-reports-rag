"""Run a gold dataset through the RAG pipeline or the agent and compare configurations.

scope="shared": search all reports at once (harder, realistic for a company-wide assistant).
scope="per_doc": search only the report(s) the question is about (FinanceBench's "single store" setting).
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


def to_markdown(results: dict[str, dict]) -> str:
    head = "| variant | " + " | ".join(COLUMNS) + " |\n|---|" + "---|" * len(COLUMNS)
    lines = [head]
    for name, res in results.items():
        s = res["summary"]
        lines.append(f"| {name} | " + " | ".join("-" if s[c] is None else f"{s[c]:.2f}" for c in COLUMNS) + " |")
    first = next(iter(results.values()))["summary"]
    lines.append(f"\n{first['n_answerable']} answerable and {first['n_unanswerable']} unanswerable questions.")
    return "\n".join(lines)
