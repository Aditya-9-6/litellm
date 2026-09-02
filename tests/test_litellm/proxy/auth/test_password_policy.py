"""
Tests for litellm/proxy/auth/password_policy.py

The HIBP client is injected as a real AsyncHTTPHandler wrapping an
httpx.MockTransport, so no network is touched and nothing is monkeypatched.
"""

import hashlib

import httpx
import pytest
from fastapi import HTTPException

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import PasswordPolicy
from litellm.proxy.auth.password_policy import (
    PasswordBreached,
    PasswordTooShort,
    get_password_policy,
    raise_password_validation_error,
    validate_new_password,
)


def _sha1_upper(password: str) -> str:
    return hashlib.sha1(password.encode("utf-8"), usedforsecurity=False).hexdigest().upper()


def _client_with_transport(handler) -> AsyncHTTPHandler:
    http_handler = AsyncHTTPHandler()
    http_handler.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return http_handler


def _client_never_called() -> AsyncHTTPHandler:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected HTTP call to {request.url}")

    return _client_with_transport(handler)


def _client_returning(body: str, status_code: int = 200) -> AsyncHTTPHandler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text=body)

    return _client_with_transport(handler)


# ---------------------------------------------------------------------------
# get_password_policy
# ---------------------------------------------------------------------------


def test_get_password_policy_defaults_when_unset():
    policy = get_password_policy(None)
    assert policy.min_length == 12
    assert policy.check_breached_passwords is True
    assert get_password_policy({}) == policy


def test_get_password_policy_parses_raw_dict():
    policy = get_password_policy({"password_policy": {"min_length": 20, "check_breached_passwords": False}})
    assert policy.min_length == 20
    assert policy.check_breached_passwords is False


def test_get_password_policy_passes_through_parsed_model():
    parsed = PasswordPolicy(min_length=15)
    assert get_password_policy({"password_policy": parsed}) is parsed


# ---------------------------------------------------------------------------
# validate_new_password
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejects_password_below_min_length_without_network():
    failure = await validate_new_password(
        password="Eleven-char",
        policy=PasswordPolicy(),
        client=_client_never_called(),
    )
    assert failure == PasswordTooShort(min_length=12)


@pytest.mark.asyncio
async def test_skips_breach_check_when_disabled():
    failure = await validate_new_password(
        password="password12345",  # breached in reality, but the check is off
        policy=PasswordPolicy(check_breached_passwords=False),
        client=_client_never_called(),
    )
    assert failure is None


@pytest.mark.asyncio
async def test_rejects_breached_password():
    password = "correct horse battery staple"
    sha1 = _sha1_upper(password)
    body = f"AAAA000000000000000000000000000000A:0\r\n{sha1[5:]}:42\r\nBBBB000000000000000000000000000000B:7"

    failure = await validate_new_password(
        password=password,
        policy=PasswordPolicy(),
        client=_client_returning(body),
    )
    assert failure == PasswordBreached()


@pytest.mark.asyncio
async def test_only_sha1_prefix_leaves_the_proxy():
    password = "a very secret password"
    sha1 = _sha1_upper(password)
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(200, text="0000000000000000000000000000000000A:1")

    failure = await validate_new_password(
        password=password,
        policy=PasswordPolicy(),
        client=_client_with_transport(handler),
    )
    assert failure is None

    (request,) = captured_requests
    assert request.url.path == f"/range/{sha1[:5]}"
    assert sha1[5:] not in str(request.url)
    assert request.headers["Add-Padding"] == "true"
    assert "litellm" in request.headers["User-Agent"]


@pytest.mark.asyncio
async def test_ignores_padding_entries_with_zero_count():
    password = "a padded-away password"
    sha1 = _sha1_upper(password)

    failure = await validate_new_password(
        password=password,
        policy=PasswordPolicy(),
        client=_client_returning(f"{sha1[5:]}:0"),
    )
    assert failure is None


@pytest.mark.asyncio
async def test_accepts_password_absent_from_breach_corpus():
    failure = await validate_new_password(
        password="a genuinely novel password",
        policy=PasswordPolicy(),
        client=_client_returning("0018A45C4D1DEF81644B54AB7F969B88D65:1\r\n00D4F6E8FA6EECAD2A3AA415EEC418D38EC:2"),
    )
    assert failure is None


@pytest.mark.asyncio
async def test_fails_open_on_network_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    failure = await validate_new_password(
        password="password12345",  # breached, but HIBP is unreachable
        policy=PasswordPolicy(),
        client=_client_with_transport(handler),
    )
    assert failure is None


@pytest.mark.asyncio
async def test_fails_open_on_http_error_status():
    failure = await validate_new_password(
        password="password12345",
        policy=PasswordPolicy(),
        client=_client_returning("service unavailable", status_code=503),
    )
    assert failure is None


@pytest.mark.asyncio
async def test_fails_open_on_malformed_response_body():
    failure = await validate_new_password(
        password="password12345",
        policy=PasswordPolicy(),
        client=_client_returning(f"{_sha1_upper('password12345')[5:]}:not-a-number"),
    )
    assert failure is None


# ---------------------------------------------------------------------------
# raise_password_validation_error
# ---------------------------------------------------------------------------


def test_too_short_maps_to_400_naming_the_minimum():
    with pytest.raises(HTTPException) as exc_info:
        raise_password_validation_error(PasswordTooShort(min_length=14))
    assert exc_info.value.status_code == 400
    assert "at least 14 characters" in exc_info.value.detail["error"]


def test_breached_maps_to_400_naming_the_breach():
    with pytest.raises(HTTPException) as exc_info:
        raise_password_validation_error(PasswordBreached())
    assert exc_info.value.status_code == 400
    assert "data breaches" in exc_info.value.detail["error"]
