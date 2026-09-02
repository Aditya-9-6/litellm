"""
Unit tests for litellm/llms/anthropic/count_tokens/handler.py

Regression coverage for handle_count_tokens_request's URL construction:
a deployment ``api_base`` is a host-style base (or a base carrying a chat
suffix like ``/v1/messages``), never the full count-tokens URL. The handler
has to strip any chat suffix and append ``/v1/messages/count_tokens`` before
POSTing, or an official Anthropic count silently fails and the caller falls
back to the local tokenizer.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.llms.anthropic.count_tokens.handler import AnthropicCountTokensHandler


def _ok_count_response() -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json={"input_tokens": 7},
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages/count_tokens"),
    )


@pytest.fixture
def patched_client():
    fake_client = MagicMock()
    fake_client.post = AsyncMock(return_value=_ok_count_response())
    with patch(
        "litellm.llms.anthropic.count_tokens.handler.get_async_httpx_client",
        return_value=fake_client,
    ):
        yield fake_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "api_base",
    [
        "https://api.anthropic.com",
        "https://api.anthropic.com/",
        "https://api.anthropic.com/v1",
        "https://api.anthropic.com/v1/messages",
        "https://api.anthropic.com/v1/messages/",
    ],
)
async def test_api_base_variants_post_to_count_tokens_endpoint(patched_client, api_base):
    handler = AnthropicCountTokensHandler()

    await handler.handle_count_tokens_request(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "hi"}],
        api_key="sk-ant-test",
        api_base=api_base,
    )

    patched_client.post.assert_awaited_once()
    (posted_url,) = patched_client.post.call_args.args
    assert posted_url == "https://api.anthropic.com/v1/messages/count_tokens"


@pytest.mark.asyncio
async def test_missing_api_base_falls_back_to_default_endpoint(patched_client):
    handler = AnthropicCountTokensHandler()

    await handler.handle_count_tokens_request(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "hi"}],
        api_key="sk-ant-test",
        api_base=None,
    )

    patched_client.post.assert_awaited_once()
    (posted_url,) = patched_client.post.call_args.args
    assert posted_url == "https://api.anthropic.com/v1/messages/count_tokens"
