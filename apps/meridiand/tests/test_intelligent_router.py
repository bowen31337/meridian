"""Tests for the ported openclaw intelligent-router tier classifier."""

from __future__ import annotations

from meridiand._intelligent_router import TIERS, classify_tier
import pytest


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        # SIMPLE — monitoring, status, lookups, small talk.
        ("show me the status", "simple"),
        ("check system status and report any anomalies", "simple"),
        ("hi", "simple"),
        ("what is the current time", "simple"),
        # CRITICAL — multiple security/production keywords dominate.
        ("audit the production payment system for security vulnerabilities", "critical"),
        # REASONING — explicit proof/formal-logic markers.
        ("prove the theorem rigorously and derive each lemma step by step", "reasoning"),
    ],
)
def test_classifications(task: str, expected: str) -> None:
    assert classify_tier(task) == expected


def test_returns_a_known_tier_for_arbitrary_input() -> None:
    for task in ["", "asdf", "build and deploy the service then run the tests", "write a poem"]:
        assert classify_tier(task) in TIERS


def test_architecture_task_is_at_least_complex() -> None:
    tier = classify_tier(
        "design a scalable distributed microservices architecture with fault tolerance "
        "and an api gateway across 5 services"
    )
    assert tier in ("complex", "critical", "reasoning")


def test_single_critical_keyword_floors_at_complex() -> None:
    # One critical keyword shouldn't force critical, but lifts it off simple/medium.
    tier = classify_tier("review the security of this login flow")
    assert tier in ("complex", "reasoning", "critical")


def test_agentic_task_floors_at_medium() -> None:
    # A short agentic task (run/test) with little else scores low -> floored to medium.
    assert classify_tier("run the tests") == "medium"


def test_extremely_dense_task_tops_out_critical() -> None:
    # A long, signal-saturated task (no critical keywords) reaches the top tier.
    core = (
        "implement and refactor the python rust go java code in main.py app.js lib.go: "
        "first build the API then integrate the kubernetes docker graphql grpc oauth redis "
        "kafka rabbitmq websocket pipeline; you must exactly precisely specifically optimize "
        "evaluate architect design and enhance the distributed scalable fault tolerant "
        "microservices service mesh architecture with high availability load balancing event "
        "driven message queue across 15 services. how why what should the json table markdown "
        "output be? verify the design. "
    )
    task = core + ("data value result item field record entry " * 90)
    assert classify_tier(task) == "critical"


def test_count_falls_back_to_keyword_on_bad_regex() -> None:
    from meridiand._intelligent_router import _count

    # An invalid regex pattern degrades to a literal keyword count.
    assert _count("a(b literal a(b", ["a(b"], regex=True) == 2
