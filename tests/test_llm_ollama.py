import json

import httpx
import pytest

from safeadapt.adapters.llm_ollama import OllamaRespondentSimulator, OllamaToMEstimator
from safeadapt.domain.models import DistressProfile, Persona, Probe


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Each `post()` pops the next canned item from a shared queue, regardless
    of which `httpx.AsyncClient(...)` instance made the call — mirrors the
    adapter creating a fresh client per retry attempt."""

    def __init__(self, queue):
        self._queue = queue

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def post(self, url, json):
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _patch_client(monkeypatch, queue):
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeAsyncClient(queue))


async def test_respond_returns_stripped_text(monkeypatch):
    _patch_client(monkeypatch, [_FakeResponse({"message": {"content": "  I'm doing alright.  "}})])
    sim = OllamaRespondentSimulator(model="qwen2.5:7b-instruct")
    profile = DistressProfile(label="low", phq9_true_severity="low", climate_true_severity="low")
    probe = Probe(item_id="p1", text="How have you been feeling?", kind="fixed_core")
    text = await sim.respond(profile=profile, persona=Persona.PEER, probe=probe, history=[])
    assert text == "I'm doing alright."


async def test_estimate_parses_and_clamps_structured_output(monkeypatch):
    payload = {
        "reasoning": "Hedged language suggests more distress than stated.",
        "beliefs": "Likely downplaying out of shame.",
        "affect": 1.5,  # out of range — must clamp
        "epistemic_certainty": 0.6,
        "social_bias": -0.2,  # out of range — must clamp
    }
    _patch_client(monkeypatch, [_FakeResponse({"message": {"content": json.dumps(payload)}})])
    tom = OllamaToMEstimator(model="qwen2.5:7b-instruct")
    probe = Probe(item_id="p1", text="How have you been feeling?", kind="fixed_core")
    state = await tom.estimate(persona=Persona.RESEARCHER, history=[(probe, "I'm fine, I guess.")])
    assert state.affect == 1.0
    assert state.social_bias == 0.0
    assert state.epistemic_certainty == 0.6
    assert "downplaying" in state.beliefs


async def test_estimate_retries_after_malformed_json_then_succeeds(monkeypatch):
    good = {"reasoning": "ok", "beliefs": "b", "affect": 0.4, "epistemic_certainty": 0.5, "social_bias": 0.3}
    queue = [
        _FakeResponse({"message": {"content": "not valid json"}}),
        _FakeResponse({"message": {"content": json.dumps(good)}}),
    ]
    _patch_client(monkeypatch, queue)
    monkeypatch.setattr("asyncio.sleep", lambda *_a, **_k: _noop())
    tom = OllamaToMEstimator()
    probe = Probe(item_id="p1", text="q", kind="fixed_core")
    state = await tom.estimate(persona=None, history=[(probe, "a")])
    assert state.affect == 0.4
    assert queue == []  # both canned responses were consumed


async def test_project_raises_after_exhausting_retries(monkeypatch):
    queue = [httpx.ConnectError("down")] * 3
    _patch_client(monkeypatch, queue)
    monkeypatch.setattr("asyncio.sleep", lambda *_a, **_k: _noop())
    tom = OllamaToMEstimator()
    probe = Probe(item_id="p1", text="q", kind="fixed_core")
    candidate = Probe(item_id="p2", text="follow-up", kind="adaptive_probe")
    with pytest.raises(RuntimeError):
        await tom.project(persona=Persona.PEER, history=[(probe, "a")], candidate=candidate)


async def _noop():
    return None
