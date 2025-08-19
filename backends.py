from __future__ import annotations
from typing import List, Optional, Any
from agent import ChatBackend, ChatMessage, StreamCallback, AgentConfig

# -----------------------
# OpenAI chat backend (optional; you can ignore if only using Ollama)
# -----------------------
class OpenAIBackend(ChatBackend):
    def __init__(self, api_key: Optional[str] = None):
        import os
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def chat(
        self,
        model: str,
        messages: List[ChatMessage],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        on_delta: Optional[StreamCallback] = None,
        **kwargs: Any,
    ) -> str:
        oai_msgs = [{"role": m.role, "content": m.content} for m in messages]
        resp = self.client.chat.completions.create(
            model=model,
            messages=oai_msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            **kwargs,
        )
        return resp.choices[0].message.content or ""

# -----------------------
# Ollama backend (local, open-source models)
# -----------------------
class OllamaBackend(ChatBackend):
    """
    Minimal Ollama chat backend using the local HTTP API.
    Docs: https://github.com/ollama/ollama/blob/main/docs/api.md
    """
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _to_ollama_msgs(self, messages: List[ChatMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def chat(
        self,
        model: str,
        messages: List[ChatMessage],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        on_delta: Optional[StreamCallback] = None,
        **kwargs: Any,
    ) -> str:
        import requests, json
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._to_ollama_msgs(messages),
            "stream": False,  # simpler integration for Gradio demo
            "options": {"temperature": temperature},
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        if "options" in kwargs and isinstance(kwargs["options"], dict):
            payload["options"].update(kwargs["options"])
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message") or {}
        if "content" in msg:
            return msg["content"]
        return data.get("response", "")

# -----------------------
# Factory helper
# -----------------------
def make_backend(cfg: AgentConfig) -> ChatBackend:
    if cfg.provider == "openai":
        return OpenAIBackend()
    if cfg.provider == "ollama":
        return OllamaBackend(base_url=cfg.ollama_base_url)
    raise ValueError(f"Unknown provider: {cfg.provider}")
