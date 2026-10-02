"""The only path from the health domain to a model API.

`send_remote` always runs the egress guard over every request before it reaches the provider
(`providers.anthropic.send`, passed the guard as its `check`). This is the one module in the health
domain allowed to import `providers` (a test enforces that). What the guard sees was already scrubbed by
the core's `Boundary.scrub`, which `runner.make_boundary` always sets for a remote turn. If the guard
raises, no client is created and no request is made.
"""
from __future__ import annotations

from harness.model import Executor, ModelResult, Sender
from harness.router import Route
from providers import anthropic as provider

from .egress import Guard


def send_remote(system: str, prompt: str, route: Route, executor: Executor, guard: Guard) -> ModelResult:
    """Guard, then send. A blocked call comes back as stop "blocked" with nothing sent."""
    return provider.send(system, prompt, route, executor, check=guard.check_request)


def make_sender(guard: Guard) -> Sender:
    """The `sender` the core calls for a live turn."""
    return lambda system, prompt, route, executor: send_remote(system, prompt, route, executor, guard)
