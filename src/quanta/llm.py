"""Model backends.

The harness talks to models through one small protocol so the same agent
code runs against Claude (`AnthropicModel`) or a deterministic test double
(`ScriptedModel`). Conversation history is append-only: assistant content is
passed back exactly as returned (thinking blocks included), and long
horizons are handled by starting fresh, state-seeded episodes rather than by
editing old turns.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

DEFAULT_MODEL = "claude-opus-5-5"

# USD per million tokens (input, output). Used for budget enforcement only.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


class ModelError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens


@dataclass
class ModelResponse:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    content: list[Any]                 # append verbatim as the assistant turn
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    model: str = ""
    refusal_category: str | None = None


class Model(Protocol):
    name: str

    def respond(self, *, system: str, messages: list[dict], tools: list[dict],
                effort: str | None = None, max_tokens: int | None = None) -> ModelResponse: ...


def estimate_cost(model: str, usage: Usage) -> float:
    pin, pout = PRICING_PER_MTOK.get(model, PRICING_PER_MTOK[DEFAULT_MODEL])
    return (usage.input_tokens * pin + usage.cache_write_tokens * pin * 1.25
            + usage.cache_read_tokens * pin * 0.1 + usage.output_tokens * pout) / 1e6


def to_jsonable(content: list[Any]) -> list[dict]:
    """Serialize content blocks (SDK objects or dicts) for transcripts."""
    out = []
    for block in content:
        if isinstance(block, dict):
            out.append(block)
        elif hasattr(block, "model_dump"):
            out.append(block.model_dump(mode="json", exclude_none=True))
        else:
            out.append({"type": "unknown", "repr": repr(block)})
    return out


class AnthropicModel:
    """Claude via the official SDK, with adaptive thinking, explicit effort,
    server-side web search/fetch, prompt caching and refusal fallbacks."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        effort: str = "high",
        max_tokens: int = 16000,
        web_search: bool = True,
        web_fetch: bool = True,
        fallbacks: bool = True,
        max_pause_continuations: int = 5,
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover - depends on environment
                raise ModelError("the 'anthropic' package is required: pip install 'quanta[llm]'") from e
            client = anthropic.Anthropic()
        self.client = client
        self.name = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks
        self.max_pause_continuations = max_pause_continuations
        self.server_tools: list[dict] = []
        if web_search:
            self.server_tools.append({"type": "web_search_20260209", "name": "web_search"})
        if web_fetch:
            self.server_tools.append({"type": "web_fetch_20260209", "name": "web_fetch"})

    def _create(self, **params):
        try:
            import anthropic
        except ImportError:  # pragma: no cover
            anthropic = None
        try:
            return self.client.beta.messages.create(**params)
        except Exception as e:
            if anthropic is not None and isinstance(e, anthropic.APIStatusError):
                raise ModelError(f"API error {e.status_code}: {e.message}") from e
            if anthropic is not None and isinstance(e, anthropic.APIConnectionError):
                raise ModelError(f"connection error: {e}") from e
            raise

    def respond(self, *, system: str, messages: list[dict], tools: list[dict],
                effort: str | None = None, max_tokens: int | None = None) -> ModelResponse:
        params: dict[str, Any] = {
            "model": self.name,
            "max_tokens": max_tokens or self.max_tokens,
            # Stable prefix first (tools -> system), cached; the tail is auto-cached.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort or self.effort},
            "cache_control": {"type": "ephemeral"},
        }
        if tools or self.server_tools:
            params["tools"] = [*tools, *self.server_tools]
        if self.fallbacks:
            params["betas"] = ["server-side-fallback-2026-07-01"]
            params["fallbacks"] = "default"
        convo = list(messages)
        content: list[Any] = []
        usage = Usage()
        resp = None
        for _ in range(self.max_pause_continuations + 1):
            resp = self._create(messages=convo, **params)
            content.extend(resp.content)
            u = resp.usage
            usage.input_tokens += getattr(u, "input_tokens", 0) or 0
            usage.output_tokens += getattr(u, "output_tokens", 0) or 0
            usage.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
            usage.cache_write_tokens += getattr(u, "cache_creation_input_tokens", 0) or 0
            if resp.stop_reason != "pause_turn":
                break
            # A long server-tool turn paused: resend with the partial turn appended.
            convo = [*messages, {"role": "assistant", "content": list(content)}]
        text = "".join(b.text for b in content if getattr(b, "type", None) == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in content if getattr(b, "type", None) == "tool_use"]
        refusal = None
        if resp.stop_reason == "refusal" and getattr(resp, "stop_details", None):
            refusal = getattr(resp.stop_details, "category", None) or "unspecified"
        return ModelResponse(text=text, tool_calls=calls, stop_reason=resp.stop_reason or "end_turn",
                             content=content, usage=usage, cost_usd=estimate_cost(self.name, usage),
                             model=getattr(resp, "model", self.name), refusal_category=refusal)


Script = Callable[[str, list[dict], list[dict]], dict]


class ScriptedModel:
    """Deterministic model for tests and offline dry runs.

    `script` is either a list of response specs consumed in order, or a
    function (system, messages, tools) -> spec. A spec is a dict with
    optional keys `text`, `tool_calls` (list of (name, input) pairs) and
    `stop_reason`.
    """

    def __init__(self, script: list[dict] | Script, name: str = "scripted") -> None:
        self.script = script
        self.name = name
        self.calls: list[dict] = []
        self._ids = itertools.count(1)

    def respond(self, *, system: str, messages: list[dict], tools: list[dict],
                effort: str | None = None, max_tokens: int | None = None) -> ModelResponse:
        self.calls.append({"system": system, "messages": list(messages),
                           "tools": [t.get("name") for t in tools], "effort": effort})
        if callable(self.script):
            spec = self.script(system, messages, tools)
        else:
            if not self.script:
                raise ModelError("scripted model ran out of responses")
            spec = self.script.pop(0)
        content: list[dict] = []
        if spec.get("text"):
            content.append({"type": "text", "text": spec["text"]})
        calls = []
        for name, inp in spec.get("tool_calls", []):
            call = ToolCall(f"toolu_{next(self._ids)}", name, dict(inp))
            calls.append(call)
            content.append({"type": "tool_use", "id": call.id, "name": name, "input": call.input})
        stop = spec.get("stop_reason") or ("tool_use" if calls else "end_turn")
        usage = Usage(input_tokens=spec.get("input_tokens", 100), output_tokens=spec.get("output_tokens", 50))
        return ModelResponse(text=spec.get("text", ""), tool_calls=calls, stop_reason=stop,
                             content=content, usage=usage, cost_usd=0.0, model=self.name,
                             refusal_category=spec.get("refusal_category"))
