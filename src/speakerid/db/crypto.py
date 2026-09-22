"""AES-GCM encryption of embeddings at rest.

A voiceprint is biometric data under BIPA, CUBI, My Health My Data and GDPR
Art. 9. It is stored encrypted, with the key held outside the database
(CLAUDE.md §11).

Ciphertext layout::

    b"SPKR1" || nonce (12 bytes) || AES-GCM ciphertext+tag

Each record is bound to its owner through the GCM additional-authenticated-data
field, so a ciphertext lifted from one person's row cannot be pasted into
another's: decryption fails rather than silently returning someone else's
voice.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from ..config import Config
from ..errors import CryptoError
from ..nptypes import FloatArray

LOGGER = logging.getLogger(__name__)

MAGIC = b"SPKR1"
NONCE_BYTES = 12
SALT_BYTES = 16
KEY_BYTES = 32  # AES-256

#: The dtype embeddings are serialised as. Fixed so a database written by one
#: build is readable by the next.
VECTOR_DTYPE = np.float32


# ---------------------------------------------------------------------------
# Key acquisition
# ---------------------------------------------------------------------------


def _load_or_create_salt(salt_path: str) -> bytes:
    """Read the KDF salt, creating it on first use with 0600 permissions."""
    if os.path.isfile(salt_path):
        with open(salt_path, "rb") as handle:
            salt = handle.read()
        if len(salt) != SALT_BYTES:
            raise CryptoError(
                "salt file {0} is {1} bytes, expected {2}; refusing to derive a "
                "key from it".format(salt_path, len(salt), SALT_BYTES)
            )
        return salt

    parent = os.path.dirname(os.path.abspath(salt_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    salt = os.urandom(SALT_BYTES)
    # Create with restrictive permissions from the start, never chmod after:
    # a window where the file exists world-readable is a window too many.
    # O_BINARY matters on Windows: without it os.write translates every 0x0A
    # byte in the random salt into CR LF, so roughly one salt in sixteen comes
    # back a byte longer than it was written and the database it keyed becomes
    # unreadable. The flag does not exist on POSIX, hence the getattr.
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(salt_path, flags, 0o600)
    try:
        os.write(descriptor, salt)
    finally:
        os.close(descriptor)

    if os.name == "nt":
        # Windows ignores the POSIX mode bits: the file inherits the parent
        # directory's ACL instead. The salt is not itself a secret -- it
        # defeats precomputation, while the passphrase provides the strength --
        # but the difference should be visible to whoever secures the host
        # rather than silently assumed away.
        LOGGER.warning(
            "created KDF salt at %s; on Windows the POSIX mode is not enforced, "
            "so restrict the data directory with an ACL (and prefer full-disk "
            "encryption) to keep it off other accounts",
            salt_path,
        )
    else:
        LOGGER.info("created new KDF salt at %s", salt_path)
    return salt


def derive_key(passphrase: str, salt: bytes, iterations: int) -> bytes:
    """PBKDF2-HMAC-SHA256 passphrase to a 256-bit AES key."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    if not passphrase:
        raise CryptoError("refusing to derive a key from an empty passphrase")
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=KEY_BYTES, salt=salt, iterations=iterations)
    return kdf.derive(passphrase.encode("utf-8"))


def resolve_key(config: Config, passphrase: Optional[str] = None) -> bytes:
    """Obtain the encryption key from the configured source.

    Never from a file in the repository, and never with a default value: an
    unencrypted-by-accident biometric database is the failure this prevents.
    """
    source = config.get_str("crypto.key_source")
    iterations = config.get_int("crypto.kdf_iterations")
    salt = _load_or_create_salt(config.resolve_path("crypto.salt_path"))

    if source == "passphrase":
        secret = passphrase
        if secret is None:
            secret = os.environ.get(config.get_str("crypto.env_var"))
        if not secret:
            raise CryptoError(
                "no passphrase supplied. Pass --passphrase, or set the "
                "{0} environment variable. The key is never stored in the "
                "repository or the database.".format(config.get_str("crypto.env_var"))
            )
        return derive_key(secret, salt, iterations)

    if source == "env":
        secret = os.environ.get(config.get_str("crypto.env_var"))
        if not secret:
            raise CryptoError(
                "crypto.key_source is 'env' but {0} is unset".format(
                    config.get_str("crypto.env_var")
                )
            )
        return derive_key(secret, salt, iterations)

    if source == "keyring":
        try:
            import keyring  # optional dependency, only on hosts that use it
        except ImportError as exc:
            raise CryptoError(
                "crypto.key_source is 'keyring' but the keyring package is not "
                "installed: {0}".format(exc)
            ) from exc
        secret = keyring.get_password(
            config.get_str("crypto.keyring_service"), config.get_str("crypto.keyring_user")
        )
        if not secret:
            raise CryptoError("no key found in the OS keyring for this service/user")
        return derive_key(secret, salt, iterations)

    raise CryptoError("unknown crypto.key_source: {0}".format(source))


# ---------------------------------------------------------------------------
# Encryption
# ---------------------------------------------------------------------------


class EmbeddingCipher(object):
    """Encrypts and decrypts embedding vectors with AES-GCM."""

    def __init__(self, key: bytes) -> None:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        if len(key) != KEY_BYTES:
            raise CryptoError("AES key must be {0} bytes, got {1}".format(KEY_BYTES, len(key)))
        self._aesgcm = AESGCM(key)

    @classmethod
    def from_config(cls, config: Config, passphrase: Optional[str] = None) -> EmbeddingCipher:
        return cls(resolve_key(config, passphrase))

    # -- bytes ---------------------------------------------------------------

    def encrypt(self, plaintext: bytes, associated_data: bytes = b"") -> bytes:
        nonce = os.urandom(NONCE_BYTES)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext, associated_data)
        return MAGIC + nonce + ciphertext

    def decrypt(self, blob: bytes, associated_data: bytes = b"") -> bytes:
        if not blob.startswith(MAGIC):
            raise CryptoError("ciphertext is not in the expected SPKR1 format")
        nonce = blob[len(MAGIC) : len(MAGIC) + NONCE_BYTES]
        payload = blob[len(MAGIC) + NONCE_BYTES :]
        if len(nonce) != NONCE_BYTES:
            raise CryptoError("truncated ciphertext: missing nonce")
        try:
            return self._aesgcm.decrypt(nonce, payload, associated_data)
        except Exception as exc:
            # Wrong key, tampered ciphertext, or a record bound to a different
            # person. All three are security events, not data errors.
            raise CryptoError(
                "failed to authenticate ciphertext: wrong key, corrupted record, "
                "or a record belonging to a different subject ({0})".format(type(exc).__name__)
            ) from exc

    # -- vectors -------------------------------------------------------------

    def encrypt_vector(self, vector: FloatArray, associated_data: bytes = b"") -> bytes:
        array: FloatArray = np.ascontiguousarray(vector, dtype=VECTOR_DTYPE)
        if array.ndim != 1:
            raise CryptoError("embeddings must be 1-D, got shape {0}".format(array.shape))
        return self.encrypt(array.tobytes(), associated_data)

    def decrypt_vector(self, blob: bytes, associated_data: bytes = b"") -> FloatArray:
        raw = self.decrypt(blob, associated_data)
        vector: FloatArray = np.frombuffer(raw, dtype=VECTOR_DTYPE)
        # frombuffer returns a read-only view over the plaintext buffer; copy so
        # callers can operate on it normally.
        return np.array(vector, dtype=VECTOR_DTYPE, copy=True)


def subject_aad(person_id: str, model_id: str) -> bytes:
    """Additional authenticated data binding a record to person and model.

    Including the model id means an embedding from a retired model cannot be
    silently compared against vectors from a new one.
    """
    return ("person={0};model={1}".format(person_id, model_id)).encode("utf-8")
