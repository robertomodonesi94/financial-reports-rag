"""A small tool-using agent: the model emits one JSON object per turn, either a tool call or a final answer."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from .llm import ABSTAIN_TEXT, LLM, Message
from .tools import ToolBox, ToolError

# FinanceBench document ids look like 3M_2018_10K, AMCOR_2023Q2_10Q, PEPSICO_2023_8K_dated-2023-05-30
_DOC_ID = re.compile(r"\b[A-Z0-9][A-Z0-9&.-]*_\d{4}(?:Q\d)?_[A-Z0-9]+(?:_dated-[\d-]+)?\b")

SYSTEM_PROMPT = """You are a financial analyst assistant answering questions about company reports (10-K, 10-Q, 8-K, earnings). You can use tools.
Reply with exactly ONE JSON object per turn, nothing else:
  {{"tool": "<tool name>", "args": {{...}}}}                   to call a tool
  {{"final": "<answer>", "sources": ["<doc id>", ...]}}        when you are done
Rules: find the right report first (list_documents), then search inside it; use calculate for every ratio, sum or growth rate;
give figures with unit and period; cite only doc ids that appeared in tool results; keep the answer short;
if the documents do not contain the answer, use the final answer "{abstain}".
Tools:
{tools}"""


def extract_json(text: str) -> dict | None:
    """Return the first balanced {...} object found in text, or None."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None


@dataclass
class AgentResult:
    question: str
    answer: str
    finished: bool
    steps: list[dict] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)
    unsupported_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _collect_ids(obj: object, out: set[str]) -> None:
    if isinstance(obj, dict):
        if isinstance(obj.get("doc_id"), str):
            out.add(obj["doc_id"])
        for v in obj.values():
            _collect_ids(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_ids(v, out)


class Agent:
    def __init__(self, llm: LLM, toolbox: ToolBox, max_steps: int = 4, max_new_tokens: int = 400, temperature: float = 0.0):
        self.llm, self.toolbox = llm, toolbox
        self.max_steps, self.max_new_tokens, self.temperature = max_steps, max_new_tokens, temperature

    def _system(self) -> str:
        tools = "\n".join(f"- {t['name']}({json.dumps(t['args'])}): {t['description']}" for t in self.toolbox.specs())
        return SYSTEM_PROMPT.format(abstain=ABSTAIN_TEXT, tools=tools)

    def run(self, question: str) -> AgentResult:
        messages: list[Message] = [{"role": "system", "content": self._system()}, {"role": "user", "content": question}]
        steps: list[dict] = []
        seen: set[str] = set()
        for _ in range(self.max_steps):
            out = self.llm.chat(messages, max_new_tokens=self.max_new_tokens, temperature=self.temperature)
            data = extract_json(out)
            messages.append({"role": "assistant", "content": out})
            if data is None or not ("final" in data or "tool" in data):
                steps.append({"type": "invalid", "output": out[:300]})
                messages.append({"role": "user", "content": 'Invalid reply. Answer with ONE JSON object: a tool call or {"final": ..., "sources": [...]}.'})
                continue
            if "final" in data:
                answer = str(data["final"]).strip()
                cited = {str(s) for s in data.get("sources", []) if isinstance(s, (str, int))} | set(_DOC_ID.findall(answer))
                steps.append({"type": "final", "answer": answer})
                return AgentResult(question, answer, True, steps, sorted(cited), sorted(cited - seen))
            name, args = str(data["tool"]), data.get("args") if isinstance(data.get("args"), dict) else {}
            try:
                result: object = self.toolbox.run(name, args)
            except (ToolError, ValueError, TypeError) as e:
                result = {"error": str(e)}
            _collect_ids(result, seen)
            steps.append({"type": "tool", "tool": name, "args": args, "result_preview": json.dumps(result)[:300]})
            messages.append({"role": "user", "content": "Observation: " + json.dumps(result)[:3000]})
        return AgentResult(question, ABSTAIN_TEXT, False, steps)
