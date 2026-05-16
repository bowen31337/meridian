from __future__ import annotations

import os
import threading
import time

# Crockford base32 alphabet: omits I, L, O, U to avoid visual ambiguity.
_ENCODING = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

# Module-level monotonic state shared across all calls to generate_ulid().
# Protected by _LOCK for thread safety.
_LOCK = threading.Lock()
_last_ms: int = -1
_last_rand: int = 0


def _time_ms() -> int:
    return time.time_ns() // 1_000_000


def _encode(value: int, length: int) -> str:
    chars: list[str] = []
    for _ in range(length):
        chars.append(_ENCODING[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


class MonotonicUlidGenerator:
    """
    Thread-safe monotonic ULID generator. Each instance maintains independent state,
    so IDs from different instances are not guaranteed to be globally monotonic.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_ms: int = -1
        self._last_rand: int = 0

    def generate(self) -> str:
        """
        Return a 26-character Crockford base32 ULID.

        Within the same millisecond the random component is incremented by 1 to
        preserve strict monotonicity. Raises OverflowError if 2^80 IDs are
        generated in a single millisecond (astronomically unlikely).
        """
        ms = _time_ms()
        with self._lock:
            if ms <= self._last_ms:
                self._last_rand += 1
                if self._last_rand >= (1 << 80):
                    raise OverflowError(
                        "ULID random component overflow: too many IDs generated in one millisecond"
                    )
                ms = self._last_ms
            else:
                self._last_rand = int.from_bytes(os.urandom(10), "big")
                self._last_ms = ms
            rand = self._last_rand

        return _encode(ms, 10) + _encode(rand, 16)


# Process-wide default generator used by generate_ulid().
_default_generator = MonotonicUlidGenerator()


def generate_ulid() -> str:
    """Generate a monotonic, URL-safe 26-character ULID using the process-wide generator."""
    return _default_generator.generate()
