"""Challenge-response: the defence that actually stops replay attacks.

CLAUDE.md §10 is blunt about this. AASIST-L's benchmark is synthetic speech,
not a phone held up to the microphone, and a countermeasure score is one
signal rather than a guarantee. What defeats essentially all casual replay is
asking the speaker to say a number chosen a moment ago: a pre-recorded clip
cannot answer a question it has never heard.

This module owns the challenge half -- issuing a random digit string, binding
it to a session, expiring it, and consuming it exactly once. Deciding whether
the spoken audio *contains* those digits needs a small digit-keyword model,
which is a separate component. Until one is attached, verification **fails
closed**: no verifier means no pass, never an assumed pass.

Tokens are single-use. A replay of an entire prior session -- the same audio
*and* the same token -- is rejected because the token is already spent.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .config import Config
from .errors import ChallengeError
from .nptypes import FloatArray

LOGGER = logging.getLogger(__name__)

#: A verifier answers: does this audio speak these digits?
DigitVerifier = Callable[[FloatArray, str], bool]


@dataclass
class Challenge:
    """An issued challenge awaiting an answer."""

    token: str
    digits: str
    issued_at: float
    expires_at: float
    attempts: int = 0
    consumed: bool = False

    def is_expired(self, now: Optional[float] = None) -> bool:
        return (now if now is not None else time.time()) >= self.expires_at

    def prompt(self) -> str:
        """What the operator displays or speaks to the user."""
        return " ".join(self.digits)


class ChallengeManager(object):
    """Issues, tracks and verifies challenges. Thread-safe."""

    def __init__(
        self,
        enabled: bool,
        num_digits: int,
        alphabet: str,
        ttl_seconds: float,
        require_pass: bool,
        max_attempts: int,
        verifier: Optional[DigitVerifier] = None,
    ) -> None:
        if num_digits <= 0:
            raise ChallengeError("challenge.num_digits must be positive")
        if len(set(alphabet)) < 2:
            raise ChallengeError("challenge.digit_alphabet needs at least two symbols")
        self.enabled = enabled
        self.num_digits = num_digits
        self.alphabet = alphabet
        self.ttl_seconds = ttl_seconds
        self.require_pass = require_pass
        self.max_attempts = max_attempts
        self.verifier = verifier
        self._challenges: Dict[str, Challenge] = {}
        self._lock = threading.Lock()

        if enabled and verifier is None and require_pass:
            LOGGER.warning(
                "challenge-response is enabled with no digit verifier attached; "
                "verification will fail closed. Attach a verifier before "
                "putting this in front of a door."
            )

    @classmethod
    def from_config(
        cls, config: Config, verifier: Optional[DigitVerifier] = None
    ) -> ChallengeManager:
        return cls(
            enabled=config.get_bool("challenge.enabled"),
            num_digits=config.get_int("challenge.num_digits"),
            alphabet=config.get_str("challenge.digit_alphabet"),
            ttl_seconds=config.get_float("challenge.ttl_seconds"),
            require_pass=config.get_bool("challenge.require_pass"),
            max_attempts=config.get_int("challenge.max_attempts"),
            verifier=verifier,
        )

    # -- issuing -------------------------------------------------------------

    def issue(self, now: Optional[float] = None) -> Challenge:
        """Mint a fresh challenge from a cryptographic RNG.

        ``secrets`` rather than ``random``: a predictable challenge sequence
        would let an attacker pre-record the next prompt.
        """
        moment = now if now is not None else time.time()
        digits = "".join(secrets.choice(self.alphabet) for _ in range(self.num_digits))
        challenge = Challenge(
            token=uuid.uuid4().hex,
            digits=digits,
            issued_at=moment,
            expires_at=moment + self.ttl_seconds,
        )
        with self._lock:
            self._expire_locked(moment)
            self._challenges[challenge.token] = challenge
        # The digits are deliberately not logged: a log reader must not be
        # able to reconstruct a live challenge.
        LOGGER.info("issued challenge %s (ttl %.1fs)", challenge.token, self.ttl_seconds)
        return challenge

    def get(self, token: str) -> Challenge:
        with self._lock:
            challenge = self._challenges.get(token)
        if challenge is None:
            raise ChallengeError("unknown or already-consumed challenge token")
        return challenge

    # -- verifying -----------------------------------------------------------

    def verify(self, token: str, speech: FloatArray, now: Optional[float] = None) -> bool:
        """Check the spoken audio against the issued digits, then consume.

        Returns False (never raises) for a wrong answer, so the caller can
        offer another attempt. Raises only for a token that should not have
        been presented at all: unknown, expired, spent, or out of attempts.
        """
        moment = now if now is not None else time.time()
        with self._lock:
            challenge = self._challenges.get(token)
            if challenge is None:
                raise ChallengeError("unknown or already-consumed challenge token")
            if challenge.consumed:
                # This is the replay-of-a-whole-session case.
                del self._challenges[token]
                raise ChallengeError("challenge token has already been used")
            if challenge.is_expired(moment):
                del self._challenges[token]
                raise ChallengeError(
                    "challenge expired after {0:.1f}s; issue a new one".format(self.ttl_seconds)
                )
            challenge.attempts += 1
            attempts = challenge.attempts
            digits = challenge.digits
            if attempts > self.max_attempts:
                del self._challenges[token]
                raise ChallengeError("challenge exceeded {0} attempts".format(self.max_attempts))

        if self.verifier is None:
            # Fail closed. An unverifiable challenge is a failed challenge.
            LOGGER.warning("no digit verifier attached; challenge %s fails closed", token)
            self._consume(token)
            return False

        passed = bool(self.verifier(speech, digits))
        if passed:
            self._consume(token)
        LOGGER.info("challenge %s attempt %d: %s", token, attempts, "pass" if passed else "fail")
        return passed

    def _consume(self, token: str) -> None:
        with self._lock:
            challenge = self._challenges.get(token)
            if challenge is not None:
                challenge.consumed = True
                del self._challenges[token]

    def resolve(self, token: Optional[str], speech: FloatArray) -> Optional[bool]:
        """Work out what to report to ``decide()`` for this request.

        Returns True (passed), False (failed -> REJECTED), or None (challenge
        not in use, so the decision does not consider it).
        """
        if not self.enabled:
            return None
        if token is None:
            if self.require_pass:
                LOGGER.warning(
                    "challenge-response is required but no token was supplied; "
                    "treating as a failed challenge"
                )
                return False
            return None
        return self.verify(token, speech)

    # -- housekeeping --------------------------------------------------------

    def _expire_locked(self, now: float) -> None:
        stale = [token for token, item in self._challenges.items() if item.is_expired(now)]
        for token in stale:
            del self._challenges[token]

    def purge_expired(self, now: Optional[float] = None) -> int:
        moment = now if now is not None else time.time()
        with self._lock:
            before = len(self._challenges)
            self._expire_locked(moment)
            return before - len(self._challenges)

    @property
    def outstanding(self) -> List[str]:
        with self._lock:
            return list(self._challenges)
