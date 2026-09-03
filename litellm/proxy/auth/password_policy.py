"""
Password-acceptance rules for UI users: NIST-style minimum length plus
breached-password screening via the haveibeenpwned.com k-anonymity range API.

Only the first 5 characters of the password's SHA-1 hash are ever sent to HIBP;
the check fails open (allows the password) when HIBP is unreachable.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, NoReturn

from fastapi import HTTPException
from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger
from litellm._version import version
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, get_async_httpx_client
from litellm.proxy._types import PasswordPolicy
from litellm.types.llms.custom_http import httpxSpecialProvider

HIBP_RANGE_API_BASE: Final = "https://api.pwnedpasswords.com/range"
HIBP_TIMEOUT_SECONDS: Final = 5.0


@dataclass(frozen=True, slots=True)
class PasswordTooShort:
    min_length: int


@dataclass(frozen=True, slots=True)
class PasswordBreached:
    pass


PasswordValidationFailure = PasswordTooShort | PasswordBreached


def get_password_policy(general_settings: Mapping[str, object] | None) -> PasswordPolicy:
    raw: Final = general_settings.get("password_policy") if general_settings is not None else None
    if isinstance(raw, PasswordPolicy):
        return raw
    if raw is None:
        return PasswordPolicy()
    return PasswordPolicy.model_validate(raw)


def _hibp_client() -> AsyncHTTPHandler:
    return get_async_httpx_client(
        llm_provider=httpxSpecialProvider.PasswordBreachCheck,
        params={"timeout": HIBP_TIMEOUT_SECONDS},
    )


def _matched_breach_counts(response_body: str, hash_suffix: str) -> tuple[int, ...]:
    parsed_lines: Final = (line.strip().partition(":") for line in response_body.upper().splitlines())
    return tuple(int(count.strip() or "0") for entry_suffix, _, count in parsed_lines if entry_suffix == hash_suffix)


async def _is_password_breached(password: str, client: AsyncHTTPHandler) -> bool:
    sha1_hex: Final = hashlib.sha1(password.encode("utf-8"), usedforsecurity=False).hexdigest().upper()
    try:
        response: Final = await client.get(
            f"{HIBP_RANGE_API_BASE}/{sha1_hex[:5]}",
            headers={"Add-Padding": "true", "User-Agent": f"litellm-proxy/{version}"},
        )
        response.raise_for_status()
        breach_counts: Final = _matched_breach_counts(response.text, sha1_hex[5:])
    except Exception as e:
        verbose_proxy_logger.warning("Breached-password check skipped, HIBP lookup failed: %s", e)
        return False
    return any(count > 0 for count in breach_counts)


async def validate_new_password(
    password: str,
    policy: PasswordPolicy,
    client: AsyncHTTPHandler | None = None,
) -> PasswordValidationFailure | None:
    if len(password) < policy.min_length:
        return PasswordTooShort(min_length=policy.min_length)
    if not policy.check_breached_passwords:
        return None
    if await _is_password_breached(password, client if client is not None else _hibp_client()):
        return PasswordBreached()
    return None


def raise_password_validation_error(failure: PasswordValidationFailure) -> NoReturn:
    match failure:
        case PasswordTooShort(min_length=min_length):
            raise HTTPException(
                status_code=400,
                detail={"error": f"Password must be at least {min_length} characters long."},
            )
        case PasswordBreached():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": (
                        "This password appears in known data breaches and cannot be used. "
                        "Please choose a different password."
                    )
                },
            )
        case _:
            assert_never(failure)
