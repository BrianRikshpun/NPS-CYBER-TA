# agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union

from backends import make_backend


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class AgentConfig:
    provider: str                 # "ollama" | "openai"
    model: str
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    ollama_base_url: str = "http://127.0.0.1:11434"   # NEW


def _to_openai_message(m: Union[ChatMessage, Dict[str, Any]]) -> Dict[str, str]:
    if isinstance(m, dict):
        return {"role": str(m.get("role")), "content": str(m.get("content"))}
    return {"role": m.role, "content": m.content}


def _normalize_context(context: Optional[List[Union[ChatMessage, Dict[str, Any]]]]) -> List[Dict[str, str]]:
    if not context:
        return []
    return [_to_openai_message(m) for m in context]


class Agent:
    def __init__(self, backend, cfg: AgentConfig):
        self.backend = backend
        self.cfg = cfg

    def chat(
        self,
        user_text: str,
        context: Optional[List[Union[ChatMessage, Dict[str, Any]]]] = None,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        messages: List[Dict[str, str]] = []

        if self.cfg.system_prompt:
            messages.append({"role": "system", "content": self.cfg.system_prompt})

        messages.extend(_normalize_context(context))
        messages.append({"role": "user", "content": user_text})

        opts: Dict[str, Any] = {}
        if self.cfg.provider == "openai":
            if self.cfg.temperature is not None:
                opts["temperature"] = float(self.cfg.temperature)
            if self.cfg.max_tokens is not None:
                opts["max_tokens"] = int(self.cfg.max_tokens)
        elif self.cfg.provider == "ollama":
            if self.cfg.temperature is not None:
                opts["temperature"] = float(self.cfg.temperature)
            if self.cfg.max_tokens is not None:
                opts["num_predict"] = int(self.cfg.max_tokens)

        if options:
            opts.update(options)

        return self.backend.chat(messages=messages, stream=stream, options=opts)


def build_agent(cfg: AgentConfig):
    backend = make_backend(cfg)
    return Agent(backend, cfg)
