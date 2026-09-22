"""Encryption of embeddings at rest (CLAUDE.md §11)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from speakerid.db.crypto import (
    KEY_BYTES,
    MAGIC,
    SALT_BYTES,
    EmbeddingCipher,
    derive_key,
    resolve_key,
    subject_aad,
)
from speakerid.errors import CryptoError


@pytest.fixture
def key() -> bytes:
    return derive_key("correct horse battery staple", b"0123456789abcdef", 1000)


def test_round_trip_preserves_the_vector(key):
    cipher = EmbeddingCipher(key)
    vector = np.random.RandomState(0).randn(192).astype(np.float32)
    restored = cipher.decrypt_vector(cipher.encrypt_vector(vector))
    assert np.array_equal(restored, vector)
    assert restored.dtype == np.float32


def test_ciphertext_does_not_contain_the_plaintext(key):
    cipher = EmbeddingCipher(key)
    vector = np.ones(192, dtype=np.float32)
    blob = cipher.encrypt_vector(vector)
    assert vector.tobytes() not in blob
    assert blob.startswith(MAGIC)


def test_nonce_differs_between_encryptions(key):
    """A reused GCM nonce is catastrophic; each record must get a fresh one."""
    cipher = EmbeddingCipher(key)
    vector = np.ones(192, dtype=np.float32)
    assert cipher.encrypt_vector(vector) != cipher.encrypt_vector(vector)


def test_wrong_key_fails_to_authenticate(key):
    blob = EmbeddingCipher(key).encrypt_vector(np.ones(8, dtype=np.float32))
    other = derive_key("a different passphrase", b"0123456789abcdef", 1000)
    with pytest.raises(CryptoError, match="authenticate"):
        EmbeddingCipher(other).decrypt_vector(blob)


def test_tampered_ciphertext_is_detected(key):
    cipher = EmbeddingCipher(key)
    blob = bytearray(cipher.encrypt_vector(np.ones(8, dtype=np.float32)))
    blob[-1] ^= 0xFF
    with pytest.raises(CryptoError, match="authenticate"):
        cipher.decrypt_vector(bytes(blob))


def test_record_is_bound_to_its_subject(key):
    """A ciphertext moved between people must not decrypt.

    Without this, swapping two rows in the database would silently swap two
    people's identities.
    """
    cipher = EmbeddingCipher(key)
    vector = np.ones(8, dtype=np.float32)
    blob = cipher.encrypt_vector(vector, subject_aad("alice", "model-1"))

    assert np.array_equal(cipher.decrypt_vector(blob, subject_aad("alice", "model-1")), vector)
    with pytest.raises(CryptoError, match="different subject"):
        cipher.decrypt_vector(blob, subject_aad("bob", "model-1"))


def test_record_is_bound_to_its_model(key):
    """Embeddings from different models are not comparable, so not readable."""
    cipher = EmbeddingCipher(key)
    blob = cipher.encrypt_vector(np.ones(8, dtype=np.float32), subject_aad("alice", "model-1"))
    with pytest.raises(CryptoError):
        cipher.decrypt_vector(blob, subject_aad("alice", "model-2"))


def test_malformed_blob_is_rejected(key):
    with pytest.raises(CryptoError, match="SPKR1"):
        EmbeddingCipher(key).decrypt(b"not a ciphertext at all")


def test_key_must_be_the_right_length():
    with pytest.raises(CryptoError, match="32 bytes"):
        EmbeddingCipher(b"too short")


def test_empty_passphrase_is_refused():
    with pytest.raises(CryptoError, match="empty passphrase"):
        derive_key("", b"0123456789abcdef", 1000)


def test_derived_key_length(key):
    assert len(key) == KEY_BYTES


def test_same_passphrase_and_salt_derive_the_same_key():
    first = derive_key("secret", b"0123456789abcdef", 1000)
    second = derive_key("secret", b"0123456789abcdef", 1000)
    assert first == second


def test_different_salt_derives_a_different_key():
    first = derive_key("secret", b"0123456789abcdef", 1000)
    second = derive_key("secret", b"fedcba9876543210", 1000)
    assert first != second


# ---------------------------------------------------------------------------
# Key acquisition
# ---------------------------------------------------------------------------


def test_missing_passphrase_is_refused_not_defaulted(base_config, monkeypatch):
    """There is no default key. An unencrypted-by-accident database is the
    failure this prevents."""
    monkeypatch.delenv("SPEAKERID_KEY", raising=False)
    with pytest.raises(CryptoError, match="no passphrase supplied"):
        resolve_key(base_config, passphrase=None)


def test_passphrase_comes_from_the_environment(base_config, monkeypatch):
    monkeypatch.setenv("SPEAKERID_KEY", "from-the-environment")
    assert len(resolve_key(base_config)) == KEY_BYTES


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows ignores POSIX mode bits; the salt inherits the directory ACL "
    "instead, which crypto._load_or_create_salt warns about at runtime",
)
def test_salt_file_is_created_with_restrictive_permissions(base_config, monkeypatch):
    monkeypatch.setenv("SPEAKERID_KEY", "abc")
    resolve_key(base_config)
    salt_path = base_config.resolve_path("crypto.salt_path")
    assert os.path.isfile(salt_path)
    assert os.stat(salt_path).st_mode & 0o077 == 0, "salt must not be group/world readable"


def test_salt_file_is_created_on_any_platform(base_config, monkeypatch):
    """The salt must exist and be the right size everywhere, Windows included."""
    monkeypatch.setenv("SPEAKERID_KEY", "abc")
    resolve_key(base_config)
    salt_path = base_config.resolve_path("crypto.salt_path")
    assert os.path.isfile(salt_path)
    with open(salt_path, "rb") as handle:
        assert len(handle.read()) == SALT_BYTES


def test_salt_is_stable_across_calls(base_config, monkeypatch):
    monkeypatch.setenv("SPEAKERID_KEY", "abc")
    assert resolve_key(base_config) == resolve_key(base_config)


def test_unknown_key_source_is_rejected(base_config, monkeypatch):
    from speakerid.config import load_config

    config = load_config(
        overrides={
            "crypto": {
                "key_source": "sticky_note",
                "salt_path": base_config.get_str("crypto.salt_path"),
            }
        }
    )
    with pytest.raises(CryptoError, match="unknown crypto.key_source"):
        resolve_key(config)
