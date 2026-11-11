"""LLM backends behind one tiny interface: local Hugging Face model, any OpenAI-compatible server, or a fake for tests."""
from __future__ import annotations

from typing import Callable, Protocol

import requests

from .config import Config

Message = dict[str, str]
ABSTAIN_TEXT = "I don't know based on the provided documents."


class LLM(Protocol):
    def chat(self, messages: list[Message], max_new_tokens: int = 400, temperature: float = 0.0) -> str: ...


class FakeLLM:
    """Scripted LLM for unit tests and dry runs (no GPU, no download)."""

    def __init__(self, responses: list[str] | None = None, fn: Callable[[list[Message]], str] | None = None):
        self.responses = list(responses or [])
        self.fn = fn
        self.calls: list[list[Message]] = []

    def chat(self, messages: list[Message], max_new_tokens: int = 400, temperature: float = 0.0) -> str:
        self.calls.append(messages)
        if self.fn is not None:
            return self.fn(messages)
        if not self.responses:
            raise RuntimeError("FakeLLM has no scripted response left")
        return self.responses.pop(0)


class OpenAICompatLLM:
    """Talks to any server exposing /chat/completions (vLLM, Ollama, llama.cpp, ...). Lets the API image stay CPU-only."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: int = 120):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.timeout = model, timeout
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def chat(self, messages: list[Message], max_new_tokens: int = 400, temperature: float = 0.0) -> str:
        payload = {"model": self.model, "messages": messages, "max_tokens": max_new_tokens, "temperature": temperature}
        r = requests.post(self.url, json=payload, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


def fold_system(messages: list[Message]) -> list[Message]:
    """Some chat templates reject the 'system' role: merge it into the first user message."""
    system = [m["content"] for m in messages if m["role"] == "system"]
    rest = [dict(m) for m in messages if m["role"] != "system"]
    if system and rest and rest[0]["role"] == "user":
        rest[0]["content"] = "\n\n".join(system) + "\n\n" + rest[0]["content"]
    return rest


class HFChatLLM:
    """Local Hugging Face causal LM, 4-bit quantised when a GPU is available (fits a free Colab T4)."""

    def __init__(self, model_name: str, load_in_4bit: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        kwargs: dict = {"device_map": "auto"}
        if torch.cuda.is_available() and load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        else:
            kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    def chat(self, messages: list[Message], max_new_tokens: int = 400, temperature: float = 0.0) -> str:
        try:
            ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        except Exception:
            ids = self.tokenizer.apply_chat_template(fold_system(messages), add_generation_prompt=True, return_tensors="pt")
        ids = ids.to(self.model.device)
        gen: dict = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0, "pad_token_id": self.tokenizer.eos_token_id}
        if temperature > 0:
            gen["temperature"] = temperature
        with self._torch.no_grad():
            out = self.model.generate(ids, **gen)
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def get_llm(cfg: Config) -> LLM:
    if cfg.llm_backend == "hf":
        return HFChatLLM(cfg.llm_model)
    if cfg.llm_backend == "openai_compat":
        return OpenAICompatLLM(cfg.llm_base_url, cfg.llm_model)
    if cfg.llm_backend == "fake":
        return FakeLLM(fn=lambda _m: ABSTAIN_TEXT)
    raise ValueError(f"Unknown llm_backend: {cfg.llm_backend}")
