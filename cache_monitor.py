"""Per-request cache lookup reporting without Streamlit or shared global state."""

from contextlib import contextmanager
from contextvars import ContextVar


_events: ContextVar[list | None] = ContextVar("task6_cache_events", default=None)
_stage: ContextVar[tuple | None] = ContextVar("task6_cache_stage", default=None)


@contextmanager
def capture_cache_events():
    """Collect real lookups for this request, including LangChain worker calls."""
    events = []
    token = _events.set(events)
    try:
        yield events
    finally:
        _events.reset(token)


@contextmanager
def cache_stage(agent: str, iteration: int):
    token = _stage.set((agent, iteration))
    try:
        yield
    finally:
        _stage.reset(token)


def record_cache_lookup(status: str):
    events = _events.get()
    stage = _stage.get()
    if events is not None and stage is not None:
        agent, iteration = stage
        events.append({"iteration": iteration, "agent": agent, "status": status})
