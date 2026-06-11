"""Intelligent task-tier classifier.

A faithful port of the openclaw ``intelligent-router`` skill's classifier: a
deterministic, dependency-free 15-dimension weighted scorer that sorts a task
into a capability tier so the reply model can be sized to the work (don't burn a
frontier model on "hi"; don't hand a proof to a small one). No LLM call — it is
pure keyword/pattern scoring, so it is instant, free, and reproducible.

Tiers (cheapest → strongest): ``simple``, ``medium``, ``complex``,
``reasoning``, ``critical``. ``classify_tier(task)`` returns one of these; the
mapping from tier to concrete model lives in the routing config.
"""

from __future__ import annotations

import math
import re

# Weighted scoring dimensions (sum = 1.0).
_WEIGHTS = {
    "reasoning_markers": 0.18,
    "code_presence": 0.15,
    "multi_step_patterns": 0.12,
    "agentic_task": 0.10,
    "technical_terms": 0.10,
    "token_count": 0.08,
    "creative_markers": 0.05,
    "question_complexity": 0.05,
    "constraint_count": 0.04,
    "imperative_verbs": 0.03,
    "output_format": 0.03,
    "simple_indicators": 0.02,
    "domain_specificity": 0.02,
    "reference_complexity": 0.02,
    "negation_complexity": 0.01,
}

_REASONING_KEYWORDS = [
    "prove",
    "theorem",
    "proof",
    "derive",
    "derivation",
    "formal",
    "verify",
    "verification",
    "logic",
    "logical",
    "induction",
    "deduction",
    "lemma",
    "corollary",
    "axiom",
    "postulate",
    "qed",
    "step by step",
    "show that",
    "demonstrate that",
    "mathematically",
    "rigorously",
]
_CODE_KEYWORDS = [
    "lint",
    "refactor",
    "bug fix",
    "code review",
    "software",
    "application",
    "component",
    "module",
    "package",
    "library",
]
_CODE_PATTERNS = [
    r"`[^`]+`",
    r"```[\s\S]*?```",
    r"\bdef\b",
    r"\bclass\b",
    r"\bimport\b",
    r"\bfrom\b",
    r"\breturn\b",
    r"\bif\b.*:\s*$",
    r"\.py\b",
    r"\.js\b",
    r"\.java\b",
    r"\.cpp\b",
    r"\.rs\b",
    r"\.go\b",
    r"\bAPI\b",
    r"\bJSON\b",
    r"\bSQL\b",
    r"\b(python|javascript|java|rust|golang|c\+\+|typescript|ruby|php)\s+\w+",
    r"\bwrite\s+.*?(function|code|script|class|method|program)",
    r"\bcode\s+(for|to|that)",
    r"\bprogram(ming)?\b",
    r"\b(coding|development|implementation)\b",
]
_AGENTIC_KEYWORDS = [
    "run",
    "test",
    "fix",
    "deploy",
    "edit",
    "build",
    "create",
    "implement",
    "execute",
    "refactor",
    "migrate",
    "integrate",
    "setup",
    "configure",
    "install",
    "compile",
    "debug",
    "troubleshoot",
]
_MULTI_STEP_PATTERNS = [
    r"\bfirst\b.*\bthen\b",
    r"\bstep\s+\d+",
    r"\d+\.\s+\w+",
    r"\bnext\b",
    r"\bafter\s+that\b",
    r"\bfinally\b",
    r"\bsubsequently\b",
    r"\band then\b",
    r"\bfollowed by\b",
    r",\s*then\b",
    r"\bthen\s+\w+\s+it\b",
]
_SIMPLE_INDICATORS = [
    "check",
    "get",
    "fetch",
    "list",
    "show",
    "display",
    "status",
    "what is",
    "how much",
    "tell me",
    "find",
    "search",
    "summarize",
]
_TECHNICAL_TERMS = [
    "algorithm",
    "architecture",
    "optimization",
    "performance",
    "scalability",
    "database",
    "security",
    "authentication",
    "encryption",
    "protocol",
    "framework",
    "library",
    "dependency",
    "middleware",
    "endpoint",
    "microservice",
    "container",
    "docker",
    "kubernetes",
    "pipeline",
]
_CREATIVE_MARKERS = [
    "creative",
    "imaginative",
    "story",
    "poem",
    "narrative",
    "write a",
    "compose",
    "brainstorm",
    "innovative",
    "original",
    "artistic",
]
_IMPERATIVE_VERBS = [
    "analyze",
    "evaluate",
    "compare",
    "assess",
    "investigate",
    "examine",
    "review",
    "validate",
    "verify",
    "optimize",
    "improve",
    "enhance",
    "design",
    "architect",
    "plan",
    "structure",
    "model",
    "prototype",
    "audit",
    "inspect",
]
_CRITICAL_KEYWORDS = [
    "security",
    "production",
    "deploy",
    "release",
    "financial",
    "payment",
    "vulnerability",
    "exploit",
    "breach",
    "audit",
    "compliance",
    "regulatory",
    "critical",
    "urgent",
    "emergency",
    "live",
    "mainnet",
]
_ARCHITECTURE_KEYWORDS = [
    "architecture",
    "architect",
    "design system",
    "system design",
    "scalable",
    "distributed",
    "microservices",
    "service mesh",
    "high availability",
    "fault tolerant",
    "load balancing",
    "api gateway",
    "event driven",
    "message queue",
    "service oriented",
]
_CONSTRAINT_KEYWORDS = [
    "must",
    "should",
    "require",
    "need",
    "constraint",
    "limit",
    "restriction",
    "only",
    "exactly",
    "precisely",
    "specifically",
    "without",
    "except",
]

TIERS = ("simple", "medium", "complex", "reasoning", "critical")


def _count(text: str, patterns: list[str], *, regex: bool = False) -> int:
    lowered = text.lower()
    total = 0
    for pattern in patterns:
        if regex:
            try:
                total += len(re.findall(pattern, text, re.IGNORECASE | re.MULTILINE))
            except re.error:
                total += lowered.count(pattern.lower())
        else:
            total += lowered.count(pattern.lower())
    return total


def _dimension_scores(text: str) -> dict[str, float]:
    lowered = text.lower()
    s: dict[str, float] = {}

    s["reasoning_markers"] = min(_count(text, _REASONING_KEYWORDS) / 3.0, 1.0)

    code = _count(text, _CODE_PATTERNS, regex=True) + _count(text, _CODE_KEYWORDS)
    s["code_presence"] = min(code / 3.0, 1.0)

    multi = _count(text, _MULTI_STEP_PATTERNS, regex=True)
    multi += _count(
        text,
        [r"with\s+\w+[,\s]+\w+\s+and", r"across\s+\d+\s+(services|components|systems)"],
        regex=True,
    )
    s["multi_step_patterns"] = min(multi / 2.0, 1.0)

    agentic = _count(text, _AGENTIC_KEYWORDS)
    arch_verbs = _count(text, ["design", "architect", "plan", "structure"])
    if arch_verbs > 0:
        agentic += arch_verbs * 2  # architecture verbs count double
    s["agentic_task"] = min(agentic / 3.0, 1.0)

    tech = _count(text, _TECHNICAL_TERMS)
    arch = _count(text, _ARCHITECTURE_KEYWORDS)
    if arch > 0:
        tech += arch * 2  # architecture keywords count double
    s["technical_terms"] = min(tech / 4.0, 1.0)

    s["token_count"] = min((len(text.split()) * 1.3) / 1000.0, 1.0)
    s["creative_markers"] = min(_count(text, _CREATIVE_MARKERS) / 2.0, 1.0)

    qmarks = text.count("?")
    qwords = len(re.findall(r"\b(who|what|when|where|why|how)\b", lowered))
    s["question_complexity"] = min((qmarks + qwords) / 3.0, 1.0)

    s["constraint_count"] = min(_count(text, _CONSTRAINT_KEYWORDS) / 3.0, 1.0)
    s["imperative_verbs"] = min(_count(text, _IMPERATIVE_VERBS) / 2.0, 1.0)
    s["output_format"] = min(
        _count(text, [r"\bjson\b", r"\btable\b", r"\blist\b", r"\bmarkdown\b", r"\bformat\b"])
        / 2.0,
        1.0,
    )
    # Inverted: lots of simple indicators -> low score.
    s["simple_indicators"] = max(0.0, 1.0 - min(_count(text, _SIMPLE_INDICATORS) / 2.0, 1.0))

    domain = _count(text, [r"\b[A-Z]{2,}\b", r"\b\w+\.\w+\b"], regex=True)
    domain += _count(
        text,
        [
            "kubernetes",
            "docker",
            "redis",
            "kafka",
            "rabbitmq",
            "graphql",
            "grpc",
            "rest api",
            "websocket",
            "oauth",
        ],
    )
    s["domain_specificity"] = min(domain / 3.0, 1.0)

    refs = _count(text, [r"\bthe\s+\w+\s+(?:above|below|mentioned|previous)\b", r"\bthis\s+\w+\b"])
    s["reference_complexity"] = min(refs / 2.0, 1.0)

    neg = _count(text, [r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bwithout\b", r"\bexcept\b"])
    s["negation_complexity"] = min(neg / 3.0, 1.0)

    return s


def _confidence(score: float) -> float:
    return 1.0 / (1.0 + math.exp(-8.0 * (score - 0.5)))


def classify_tier(task: str) -> str:
    """Classify *task* into one of TIERS using the 15-dimension weighted scorer."""
    dims = _dimension_scores(task)
    score = sum(_WEIGHTS[d] * v for d, v in dims.items())
    confidence = _confidence(score)
    is_agentic = dims["agentic_task"] > 0.3 or dims["multi_step_patterns"] > 0.5

    # Critical keywords dominate: 2+ forces critical; a single one floors at complex.
    critical = _count(task, _CRITICAL_KEYWORDS)
    if critical >= 2:
        return "critical"
    if critical == 1 and score < 0.5:
        score = 0.5

    # Strong, explicit reasoning markers route to the reasoning tier.
    if dims["reasoning_markers"] >= 0.6 and (score >= 0.10 or confidence >= 0.30):
        return "reasoning"

    # Agentic floors: multi-step + code -> complex; otherwise at least medium.
    if is_agentic:
        if dims["multi_step_patterns"] > 0.3 and dims["code_presence"] > 0 and score < 0.5:
            score = 0.5
        elif score < 0.4:
            score = 0.4

    if score < 0.3:
        return "simple"
    if score < 0.5:
        return "medium"
    if score < 0.75:
        return "complex"
    return "critical"
