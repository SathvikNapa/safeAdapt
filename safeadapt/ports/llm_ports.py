"""Driven ports: interfaces the application core depends on. Concrete I/O
(mock, OpenAI, Anthropic) lives in adapters/ and implements these Protocols —
the core never imports an adapter directly."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from safeadapt.domain.models import DistressProfile, MentalState, Persona, Probe


@runtime_checkable
class RespondentSimulator(Protocol):
    """Simulates a respondent's reply to a probe, given their held-constant profile."""

    async def respond(
        self,
        *,
        profile: DistressProfile,
        persona: Persona | None,
        probe: Probe,
        history: list[str],
    ) -> str: ...


@runtime_checkable
class ToMEstimator(Protocol):
    """The Theory-of-Mind module (§3.3): estimates respondent mental state from
    the conversation so far, and projects it forward for anticipatory safety."""

    async def estimate(
        self,
        *,
        persona: Persona | None,
        history: list[tuple[Probe, str]],
    ) -> MentalState: ...

    async def project(
        self,
        *,
        persona: Persona | None,
        history: list[tuple[Probe, str]],
        candidate: Probe,
    ) -> float:
        """Projected affect A'(a_t) if `candidate` were asked next."""
        ...
