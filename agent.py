from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Literal, Callable, Any
from abc import ABC, abstractmethod

Role = Literal["system", "user", "assistant"]

@dataclass
class ChatMessage:
    role: Role
    content: str

StreamCallback = Callable[[str], None]

class ChatBackend(ABC):
    @abstractmethod
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
        ...

@dataclass
class AgentConfig:
    provider: Literal["openai", "ollama"]
    model: str
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = "You are a helpful teaching assistant."
    ollama_base_url: str = "http://localhost:11434"  # used only for provider="ollama"

class Agent:
    def __init__(self, backend: ChatBackend, config: AgentConfig):
        self.backend = backend
        self.config = config

    def chat(
        self,
        user_message: str,
        context: Optional[List[ChatMessage]] = None,
        stream: bool = False,
        on_delta: Optional[StreamCallback] = None,
        **kwargs: Any,
    ) -> str:
        messages: List[ChatMessage] = []
        if self.config.system_prompt:
            messages.append(ChatMessage("system", self.config.system_prompt))
        if context:
            messages.extend(context)
        messages.append(ChatMessage("user", user_message))

        return self.backend.chat(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=stream,
            on_delta=on_delta,
            **kwargs,
        )
