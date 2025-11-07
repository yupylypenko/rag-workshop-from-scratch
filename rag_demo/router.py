"""Query routing utilities for detecting malicious queries.

The router implements lightweight heuristics to catch obvious malicious intent
before the request reaches the retrieval pipeline. The implementation is
inspired by the safety guardrails recommended in the LangChain overview docs,
which suggest inserting routing/guard components ahead of model calls to protect
systems from prompt-injection and jailbreak attempts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable, List, Pattern, Tuple


DEFAULT_PATTERNS: Tuple[str, ...] = (
    r"(?i)ignore\s+previous\s+instructions",
    r"(?i)drop\s+table",
    r"(?i)truncate\s+table",
    r"(?i)rm\s+-rf",
    r"(?i)format\s+c:\\",
    r"(?i)delete\s+from",
    r"(?i)passwords?",
    r"(?i)api\s+keys?",
    r"(?i)system\s+prompt",
    r"(?i)powershell",
    r"(?i)malware",
    r"(?i)backdoor",
    r"(?i)virus",
)


@dataclass(slots=True)
class QueryRouter:
    """Rule-based router that flags potentially malicious queries.

    Attributes:
        blocked_patterns: Regex patterns that indicate malicious intent.
        max_length: Optional maximum query length. Overly long queries are often
            associated with prompt-injection attempts that try to stuff large
            instructions into the context window.
    """

    blocked_patterns: Iterable[str] = field(default_factory=lambda: DEFAULT_PATTERNS)
    max_length: int = 2048

    def __post_init__(self) -> None:
        self._compiled: List[Pattern[str]] = [re.compile(pat) for pat in self.blocked_patterns]

    def inspect(self, query: str) -> tuple[bool, str]:
        """Inspect a query and return a tuple of (is_allowed, reason).

        Args:
            query: Raw user query text.

        Returns:
            Tuple containing a boolean indicating if the query is allowed and a
            message describing the decision.
        """

        cleaned_query = query.strip()
        if not cleaned_query:
            return False, "Query is empty."

        if len(cleaned_query) > self.max_length:
            return (
                False,
                "Query rejected: exceeds maximum allowed length."
                " Please shorten your question and try again.",
            )

        for pattern in self._compiled:
            if pattern.search(cleaned_query):
                return (
                    False,
                    "Query rejected: detected potentially malicious intent"
                    " (contains disallowed instructions).",
                )

        return True, "Query accepted."


def is_query_safe(query: str, router: QueryRouter | None = None) -> tuple[bool, str]:
    """Helper that checks if a query is safe using the provided router.

    Args:
        query: User provided query.
        router: Optional pre-configured router. When omitted a default router is
            created.

    Returns:
        Tuple of (is_allowed, message)
    """

    router = router or QueryRouter()
    return router.inspect(query)
