"""The Anthropic Messages API: request shape and the live call. The only module that imports `anthropic`.

`build_request` is the request shape (system block with a cache breakpoint, the recall tool, adaptive
thinking and effort except on the small tier, the server-side refusal fallback on the large tier).
`send` is the live call with the tool loop: it matches the core's `Sender` signature, so it can be
passed as `Boundary.sender` directly. It owns the SDK client, refusal handling, error markers and usage
accounting. It does no egress logic: a domain that needs an outbound guard wraps it and passes `check`,
which is called with every request before a client is created or anything is sent.
"""
from __future__ import annotations

import sys
import time
from typing import Any, Callable

from harness.model import Executor, ModelResult, SendBlocked, SenderError
from harness.router import Route
from harness.tools import RECALL_TOOL

MAX_ROUNDS = 3
FALLBACK_BETA = "server-side-fallback-2026-07-01"

_client_factory: Callable[[], Any] | None = None     # tests replace this; production builds the real client


# --------------------------------------------------------------------- request

def apply_route(kwargs: dict[str, Any], route: Route) -> dict[str, Any]:
    """Tier-dependent options shared by the turn call and the judge call."""
    if route.tier != "small":        # Haiku 4.5 takes neither adaptive thinking nor effort
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": route.effort}
    if route.model == "claude-opus-5":   # server-side refusal fallback, routed by category
        kwargs["extra_headers"] = {"anthropic-beta": FALLBACK_BETA}
        kwargs["extra_body"] = {"fallbacks": "default"}
    return kwargs


def build_request(system: str, prompt: str, route: Route,
                  messages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The kwargs for `client.messages.create`, without sending anything."""
    kwargs: dict[str, Any] = dict(
        model=route.model,
        max_tokens=4096,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=messages if messages is not None else [{"role": "user", "content": prompt}],
        tools=[RECALL_TOOL],
        tool_choice={"type": "auto"},
    )
    return apply_route(kwargs, route)


# ------------------------------------------------------------------------ call

def _default_client():
    import anthropic    # credentials resolve from the environment or an `ant auth login` profile
    return anthropic.Anthropic()


def available() -> bool:
    """True when the SDK can be imported (callers use this for a friendly message before a live run)."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _is_sdk_error(exc: Exception) -> bool:
    if type(exc).__module__.startswith("anthropic") or isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    mod = sys.modules.get("anthropic")
    kinds = tuple(k for k in (getattr(mod, "APIStatusError", None), getattr(mod, "APIConnectionError", None))
                  if isinstance(k, type))
    return bool(kinds) and isinstance(exc, kinds)


def create(request: dict[str, Any]) -> Any:
    """One Messages API call. Raises SenderError for SDK, network and credential failures."""
    try:
        client = (_client_factory or _default_client)()
        return client.messages.create(**request)
    except ImportError as exc:
        raise SenderError("live calls need the anthropic package: pip install anthropic") from exc
    except Exception as exc:                  # APIStatusError / APIConnectionError / missing credentials
        if _is_sdk_error(exc):
            raise SenderError(f"{type(exc).__name__}: {exc}") from exc
        raise


def send(system: str, prompt: str, route: Route, executor: Executor,
         check: Callable[[dict[str, Any]], None] | None = None) -> ModelResult:
    """The live call: a tool loop of at most MAX_ROUNDS requests. `check(request)`, when given, runs
    before each request and may raise SendBlocked; the call then ends with stop "blocked" and nothing
    is sent. Failures come back as markers (stop "refusal", "error" or "blocked"), never as exceptions
    (apart from an unexpected bug in the SDK call)."""
    t0 = time.perf_counter()
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    usage = {"input_tokens": 0, "output_tokens": 0,
             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}

    def done(text: str, stop: str = "ok", error: str | None = None, rounds: int = 1) -> ModelResult:
        return ModelResult(text, "live", int((time.perf_counter() - t0) * 1000), route.model,
                           dict(usage), stop, error, rounds)

    try:
        for round_no in range(1, MAX_ROUNDS + 1):
            request = build_request(system, prompt, route, messages)
            if check is not None:
                check(request)
            resp = create(request)
            for k in usage:
                usage[k] += getattr(resp.usage, k, 0) or 0
            if resp.stop_reason == "refusal":
                return done("", "refusal", rounds=round_no)
            if resp.stop_reason != "tool_use":
                return done("".join(b.text for b in resp.content if b.type == "text"), rounds=round_no)
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                try:
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": executor(block.name, block.input)})
                except Exception as exc:  # report the failure to the model, do not crash the turn
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": f"tool error: {exc}", "is_error": True})
            messages.append({"role": "user", "content": results})
        return done("", "error", f"tool loop did not finish in {MAX_ROUNDS} rounds", MAX_ROUNDS)
    except SendBlocked as exc:
        return done("", "blocked", str(exc))
    except SenderError as exc:
        return done("", "error", str(exc))
