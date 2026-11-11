"""Tools the agent can call. Every tool validates its arguments and returns JSON-serialisable data."""
from __future__ import annotations

import ast
import operator

from .retrieve import Retriever

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class ToolError(Exception):
    """Raised for bad tool names or arguments; the message is fed back to the model."""


def safe_eval(expression: str) -> float:
    """Arithmetic only: numbers, + - * / // % ** and parentheses. No names, no calls."""
    if len(expression) > 200:
        raise ToolError("expression too long")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as e:
        raise ToolError(f"invalid expression: {e.msg}") from e

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 16:
                raise ToolError("exponent too large")
            try:
                return _BINOPS[type(node.op)](left, right)
            except ZeroDivisionError as e:
                raise ToolError("division by zero") from e
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        raise ToolError("only arithmetic is allowed")

    return ev(tree)


class ToolBox:
    def __init__(self, retriever: Retriever, docs: dict[str, dict]):
        self.retriever, self.docs = retriever, docs

    def specs(self) -> list[dict]:
        return [
            {"name": "list_documents", "description": "List the indexed reports (id, company, type, period). Optional filter by company.",
             "args": {"company": "string, optional"}},
            {"name": "search_corpus", "description": "Search the report chunks. Returns document ids, pages and snippets. "
             "Pass doc_id to search inside one report.",
             "args": {"query": "string", "doc_id": "string, optional", "k": "integer, optional (default 3, max 8)"}},
            {"name": "get_document_info", "description": "Metadata of one report: company, type, period, sector, pages.",
             "args": {"doc_id": "string"}},
            {"name": "calculate", "description": "Evaluate an arithmetic expression, e.g. a ratio or a growth rate: (1577-1373)/1373*100.",
             "args": {"expression": "string"}},
        ]

    def _doc(self, args: dict) -> str:
        doc_id = str(args.get("doc_id", "")).strip()
        if doc_id not in self.docs:
            raise ToolError(f"unknown doc_id '{doc_id}'; call list_documents to see the valid ids")
        return doc_id

    def run(self, name: str, args: dict | None) -> object:
        args = args or {}
        if name == "search_corpus":
            query = str(args.get("query", "")).strip()
            if not query:
                raise ToolError("search_corpus needs a non-empty 'query'")
            k = max(1, min(int(args.get("k", 3)), 8))
            doc_ids = [self._doc(args)] if args.get("doc_id") else None
            return [{"doc_id": h.chunk.doc_id, "page": h.chunk.page + 1, "title": h.chunk.title, "chunk_id": h.chunk.id,
                     "snippet": h.chunk.text[:600]} for h in self.retriever.search(query, k, doc_ids=doc_ids)]
        if name == "get_document_info":
            doc_id = self._doc(args)
            d = self.docs[doc_id]
            return {"doc_id": doc_id, **{k: d.get(k) for k in ("title", "company", "doc_type", "period", "sector", "pages")}}
        if name == "list_documents":
            company = str(args.get("company", "")).strip().lower()
            rows = [{"doc_id": i, "company": d.get("company"), "doc_type": d.get("doc_type"), "period": d.get("period")}
                    for i, d in self.docs.items() if not company or company in (d.get("company") or i).lower()]
            return rows[:80]
        if name == "calculate":
            return {"result": safe_eval(str(args.get("expression", "")))}
        raise ToolError(f"unknown tool '{name}'")
