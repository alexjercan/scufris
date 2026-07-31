"""The credentials themselves: the operator's password, and the machine token.

Nothing here logs a password, a hash, or the machine token.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import logging
import secrets

logger = logging.getLogger(__name__)

# scrypt cost. n=2**15 with r=8 needs ~32MB per hash, which is a fine price on a
# login that happens once a session and a real cost to an offline attacker. The
# parameters are encoded INTO the hash so they can be raised later without
# invalidating hashes that already exist.
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SALT_BYTES = 16


def hash_password(password: str) -> str:
    """Return an encoded ``scrypt`` hash of ``password``.

    Format: ``scrypt$<n>$<r>$<p>$<salt-b64>$<hash-b64>``. Parameters travel with
    the hash so verification never has to guess them.
    """
    salt = secrets.token_bytes(_SALT_BYTES)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
        maxmem=_SCRYPT_N * _SCRYPT_R * 2 * 64 + 1024 * 1024,
    )
    return "$".join(
        (
            "scrypt",
            str(_SCRYPT_N),
            str(_SCRYPT_R),
            str(_SCRYPT_P),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(derived).decode("ascii"),
        )
    )


def verify_password(password: str, encoded: str) -> bool:
    """Whether ``password`` matches the encoded hash. Never raises.

    A malformed, truncated, or foreign-format hash returns False: the failure
    mode of a corrupt credential must be "nobody gets in", not a 500 and not an
    accidental match.
    """
    try:
        scheme, raw_n, raw_r, raw_p, raw_salt, raw_hash = encoded.split("$")
        if scheme != "scrypt":
            return False
        n, r, p = int(raw_n), int(raw_r), int(raw_p)
        salt = base64.b64decode(raw_salt, validate=True)
        expected = base64.b64decode(raw_hash, validate=True)
        if not salt or not expected:
            return False
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=len(expected),
            maxmem=n * r * 2 * 64 + 1024 * 1024,
        )
    except (ValueError, TypeError, MemoryError, binascii.Error) as exc:
        logger.warning(
            "auth: stored password hash is unusable (%s)", type(exc).__name__
        )
        return False
    return hmac.compare_digest(derived, expected)


def mint_api_token() -> str:
    """A fresh per-process machine token for the app's own tool subprocesses.

    Never persisted and never configurable: it dies with the process, so there is
    no credential at rest and no rotation to get wrong. It authenticates
    subprocesses that already run with the operator's privileges, so it grants
    nothing they did not have.
    """
    return secrets.token_urlsafe(32)


def token_matches(presented: str | None, expected: str | None) -> bool:
    """Constant-time equality for a credential, total over ANY input string.

    Both arguments come straight off the wire (an ``Authorization`` or CSRF
    header), and Starlette decodes headers as latin-1, so a raw byte above 0x7F
    reaches this function as a non-ASCII ``str``. ``hmac.compare_digest`` raises
    ``TypeError`` on a non-ASCII ``str``, which would turn a garbage header from
    an UNAUTHENTICATED caller into a 500 and a traceback in the journal. Encoding
    first makes every input compare false instead of raising, and keeps the
    comparison constant-time. ``surrogatepass`` rather than ``surrogateescape``
    so even a lone surrogate encodes rather than raising - the point is that this
    function is TOTAL, not that it round-trips.
    """
    if not presented or not expected:
        return False
    return hmac.compare_digest(
        presented.encode("utf-8", "surrogatepass"),
        expected.encode("utf-8", "surrogatepass"),
    )


def bearer_token(authorization: str | None) -> str | None:
    """Extract the token from an ``Authorization: Bearer <token>`` header."""
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()
