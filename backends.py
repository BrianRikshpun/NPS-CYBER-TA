# backends.py
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import requests


def _to_openai_messages(messages: List[Union[Dict[str, Any], Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for m in messages:
        if isinstance(m, dict):
            role, content = m.get("role"), m.get("content")
        else:
            role, content = getattr(m, "role", None), getattr(m, "content", None)
        if role and content is not None:
            out.append({"role": str(role), "content": str(content)})
    return out


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    system = "\n".join([m["content"] for m in messages if m.get("role") == "system"])
    convo = []
    for m in messages:
        if m["role"] in ("user", "assistant"):
            prefix = "User:" if m["role"] == "user" else "Assistant:"
            convo.append(f"{prefix} {m['content']}")
    prompt = (f"System: {system}\n" if system else "") + "\n".join(convo) + "\nAssistant:"
    return prompt.strip()


class Backend(ABC):
    @abstractmethod
    def chat(self, messages: List[Union[Dict[str, Any], Any]], stream: bool = False, options: Optional[Dict[str, Any]] = None) -> str:
        ...


# ---------- OpenAI ----------
class OpenAIBackend(Backend):
    def __init__(self, model: str):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set for OpenAI backend.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def chat(self, messages: List[Union[Dict[str, Any], Any]], stream: bool = False, options: Optional[Dict[str, Any]] = None) -> str:
        msgs = _to_openai_messages(messages)
        resp = self.client.chat.completions.create(model=self.model, messages=msgs, stream=False, **(options or {}))
        return resp.choices[0].message.content or ""


# ---------- Ollama ----------
class OllamaBackend(Backend):
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _assert_is_ollama(self):
        # Quick health check to ensure something Ollama-like responds
        try:
            r = requests.get(f"{self.base_url}/api/version", timeout=5)
            if r.status_code == 404:
                # Some older versions might not have /api/version; try /api/tags
                r2 = requests.get(f"{self.base_url}/api/tags", timeout=5)
                if r2.status_code == 404:
                    raise RuntimeError(
                        f"No Ollama API at {self.base_url} (both /api/version and /api/tags returned 404). "
                        "Start Ollama (e.g., run 'ollama serve' or open the Ollama app), "
                        "or correct the Base URL in the UI."
                    )
        except requests.RequestException as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}. Is the server running? "
                "Try starting it with 'ollama serve' or adjust the Base URL."
            ) from e

    def chat(self, messages: List[Union[Dict[str, Any], Any]], stream: bool = False, options: Optional[Dict[str, Any]] = None) -> str:
        self._assert_is_ollama()

        options = options or {}
        msgs = _to_openai_messages(messages)

        # 1) Try /api/chat
        try:
            payload = {"model": self.model, "messages": msgs, "stream": stream, "options": options}
            r = requests.post(f"{self.base_url}/api/chat", json=payload, stream=stream, timeout=300)
            if r.status_code != 404:
                r.raise_for_status()
                if not stream:
                    data = r.json()
                    return data.get("message", {}).get("content", "")
                else:
                    out = []
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if "message" in obj and "content" in obj["message"]:
                                out.append(obj["message"]["content"])
                        except Exception:
                            pass
                    return "".join(out)
        except requests.HTTPError as e:
            if getattr(e, "response", None) is None or e.response.status_code != 404:
                raise

        # 2) Fallback to /api/generate (older Ollama)
        prompt = _messages_to_prompt(msgs)
        payload = {"model": self.model, "prompt": prompt, "stream": stream, "options": options}
        r = requests.post(f"{self.base_url}/api/generate", json=payload, stream=stream, timeout=300)
        if r.status_code == 404:
            raise RuntimeError(
                f"Ollama endpoint '/api/generate' not found at {self.base_url}. "
                "This usually means the service at that URL is NOT Ollama, or it's running on a different port. "
                "Check your Base URL and make sure Ollama is running."
            )
        r.raise_for_status()
        if not stream:
            return r.json().get("response", "")
        else:
            out = []
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "response" in obj:
                        out.append(obj["response"])
                except Exception:
                    pass
            return "".join(out)


def make_backend(cfg) -> Backend:
    provider = getattr(cfg, "provider", None)
    model = getattr(cfg, "model", None)
    if provider == "openai":
        return OpenAIBackend(model=model)
    if provider == "ollama":
        base_url = getattr(cfg, "ollama_base_url", "http://127.0.0.1:11434")
        return OllamaBackend(model=model, base_url=base_url)
    raise ValueError(f"Unknown provider: {provider}")
