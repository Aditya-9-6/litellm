"""
Tests for Milvus Vector Store
"""

import json
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.milvus.vector_stores.grpc_transformation import (
    MilvusGRPCVectorStoreConfig,
)
from litellm.llms.milvus.vector_stores.transformation import MilvusVectorStoreConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_stores import VectorStoreSearchOptionalRequestParams
from litellm.utils import ProviderConfigManager
from litellm.vector_stores import asearch as vector_store_asearch
from litellm.vector_stores import search as vector_store_search

# Mock response from actual Milvus API
MOCK_MILVUS_SEARCH_RESPONSE = {
    "code": 0,
    "cost": 6,
    "data": [
        {
            "book_id": 0,
            "book_intro_text": "abababababa_0562efee-0f1f-4b6b-9ca3-1a160f124ad8",
            "distance": 10.240219,
        },
        {
            "book_id": 1,
            "book_intro_text": "abababababa_9a13e8f3-bb1e-487f-b555-b8ae4b127243",
            "distance": 10.240219,
        },
        {
            "book_id": 2,
            "book_intro_text": "abababababa_870f47f1-23ec-4364-ad30-6d364ba8ddb5",
            "distance": 10.240219,
        },
        {
            "book_id": 1000,
            "book_intro_text": "abababababa_8ea2d76a-3fdf-49b3-8f16-a91638361bba",
            "distance": 8.531628,
        },
        {
            "book_id": 1001,
            "book_intro_text": "abababababa_24758251-e740-4183-8649-2f742f676ca0",
            "distance": 8.531628,
        },
        {
            "book_id": 1002,
            "book_intro_text": "abababababa_faa55789-220d-4ef1-b5bf-a72f2fbd061b",
            "distance": 8.531628,
        },
        {
            "book_id": 0,
            "book_intro_text": "abababababa_0562efee-0f1f-4b6b-9ca3-1a160f124ad8",
            "distance": 8.236887,
        },
        {
            "book_id": 1,
            "book_intro_text": "abababababa_9a13e8f3-bb1e-487f-b555-b8ae4b127243",
            "distance": 8.236887,
        },
        {
            "book_id": 2,
            "book_intro_text": "abababababa_870f47f1-23ec-4364-ad30-6d364ba8ddb5",
            "distance": 8.236887,
        },
    ],
    "topks": [3, 3, 3],
}
# Mock embedding response from OpenAI
MOCK_EMBEDDING_RESPONSE = MagicMock()
MOCK_EMBEDDING_RESPONSE.data = [
    {
        "embedding": [
            0.023,
            -0.019,
            0.045,
            -0.012,
            0.067,
            -0.034,
            0.089,
            -0.056,
        ]
        * 128  # Simulate 1024-dimensional embedding
    }
]


class MockPyMilvusHit(dict[str, object]):
    def get(self, key: str, default: object = None) -> object:
        if key == "entity":
            return {
                "book_intro_text": "closest result",
                "category": "reference",
            }
        return super().get(key, default)


class TestMilvusVectorStore:
    """Test Milvus Vector Store with mocked responses"""

    @pytest.mark.asyncio
    async def test_basic_search_with_mock_async(self):
        """Test basic vector search with mocked backend response (async)"""

        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_MILVUS_SEARCH_RESPONSE
        mock_response.text = json.dumps(MOCK_MILVUS_SEARCH_RESPONSE)

        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = MOCK_EMBEDDING_RESPONSE

            with patch(
                "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
                new_callable=AsyncMock,
            ) as mock_post:
                mock_post.return_value = mock_response

                # Make the search request
                response = await vector_store_asearch(
                    query="what is machine learning?",
                    vector_store_id="book_2",
                    custom_llm_provider="milvus",
                    api_base="https://in03-test.serverless.aws-eu-central-1.cloud.zilliz.com",
                    api_key="mock_milvus_api_key",
                    litellm_embedding_model="text-embedding-3-large",
                    litellm_embedding_config={
                        "api_key": "mock_openai_api_key",
                    },
                    outputFields=["book_intro_text"],
                    annsField="book_intro_vector",
                    milvus_text_field="book_intro_text",
                )

                print("Response:", json.dumps(response, indent=2, default=str))

                # Verify embedding was called with correct parameters
                mock_embedding.assert_called_once()
                embedding_call_args = mock_embedding.call_args
                assert embedding_call_args[1]["model"] == "text-embedding-3-large"
                assert embedding_call_args[1]["input"] == ["what is machine learning?"]
                assert embedding_call_args[1]["api_key"] == "mock_openai_api_key"

                # Verify the API was called
                mock_post.assert_called_once()

                # Verify the request payload
                call_args = mock_post.call_args
                print(f"call_args: {call_args}")
                print(f"call_args.kwargs: {call_args.kwargs}")

                # The post method is called with 'data' parameter (JSON string) not 'json' parameter
                request_data_str = call_args.kwargs.get("data")
                if request_data_str:
                    request_data = json.loads(request_data_str)
                else:
                    # Fallback: check for json kwarg or in args
                    request_data = call_args.kwargs.get("json")
                    if request_data is None and len(call_args.args) > 0 and isinstance(call_args.args[0], dict):
                        request_data = call_args.args[0]

                assert request_data is not None, f"Could not extract request data. Call args: {call_args}"
                print("Request data:", json.dumps(request_data, indent=2, default=str))

                # Validate request structure
                assert "collectionName" in request_data
                assert request_data["collectionName"] == "book_2"
                assert "data" in request_data
                assert isinstance(request_data["data"], list)
                assert len(request_data["data"]) == 1  # Single query vector
                assert "annsField" in request_data
                assert request_data["annsField"] == "book_intro_vector"
                assert "outputFields" in request_data
                assert request_data["outputFields"] == ["book_intro_text"]

                # Verify the URL format
                url = call_args.kwargs.get("url", "")
                assert "v2/vectordb/entities/search" in url

                # Validate the response structure (LiteLLM standard format)
                assert response is not None
                assert response["object"] == "vector_store.search_results.page"  # type: ignore
                assert "data" in response
                assert len(response["data"]) == 9  # type: ignore  # 9 results in mock response

                # Validate first result
                first_result = response["data"][0]  # type: ignore
                assert "score" in first_result
                assert first_result["score"] == 10.240219  # type: ignore
                assert "content" in first_result
                assert "attributes" in first_result

                # Validate content structure
                assert len(first_result["content"]) > 0  # type: ignore
                assert first_result["content"][0]["type"] == "text"  # type: ignore
                assert "text" in first_result["content"][0]  # type: ignore
                assert (
                    first_result["content"][0]["text"]  # type: ignore
                    == "abababababa_0562efee-0f1f-4b6b-9ca3-1a160f124ad8"
                )

                # Validate attributes contain book_id but NOT book_intro_text (it's in content)
                assert "book_id" in first_result["attributes"]  # type: ignore
                assert first_result["attributes"]["book_id"] == 0  # type: ignore
                assert "book_intro_text" not in first_result["attributes"]  # type: ignore  # Should be in content, not attributes

    def test_basic_search_with_mock_sync(self):
        """Test basic vector search with mocked backend response (sync)"""

        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_MILVUS_SEARCH_RESPONSE
        mock_response.text = json.dumps(MOCK_MILVUS_SEARCH_RESPONSE)

        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = MOCK_EMBEDDING_RESPONSE

            with patch(  # test-quality-ok: isolates Milvus REST request transformation from network transport
                "litellm.llms.custom_httpx.http_handler.HTTPHandler.post"
            ) as mock_post:
                mock_post.return_value = mock_response

                # Make the search request
                response = vector_store_search(
                    query="what is machine learning?",
                    vector_store_id="book_2",
                    custom_llm_provider="milvus",
                    api_base="https://in03-test.serverless.aws-eu-central-1.cloud.zilliz.com",
                    api_key="mock_milvus_api_key",
                    litellm_embedding_model="text-embedding-3-large",
                    litellm_embedding_config={
                        "api_key": "mock_openai_api_key",
                    },
                    outputFields=["book_intro_text"],
                    annsField="book_intro_vector",
                    milvus_text_field="book_intro_text",
                )

                print("Response:", json.dumps(response, indent=2, default=str))

                # Verify embedding was called
                mock_embedding.assert_called_once()

                # Verify the API was called
                mock_post.assert_called_once()

                # Verify the request payload
                call_args = mock_post.call_args

                # The post method is called with 'data' parameter (JSON string) not 'json' parameter
                request_data_str = call_args.kwargs.get("data")
                if request_data_str:
                    request_data = json.loads(request_data_str)
                else:
                    # Fallback: check for json kwarg or in args
                    request_data = call_args.kwargs.get("json")
                    if request_data is None and len(call_args.args) > 0 and isinstance(call_args.args[0], dict):
                        request_data = call_args.args[0]

                assert request_data is not None, f"Could not extract request data. Call args: {call_args}"

                # Validate request structure
                assert "collectionName" in request_data
                assert request_data["collectionName"] == "book_2"
                assert "data" in request_data
                assert isinstance(request_data["data"], list)
                assert "annsField" in request_data
                assert "outputFields" in request_data

                # Validate the response structure
                assert response is not None
                assert response["object"] == "vector_store.search_results.page"  # type: ignore
                assert "data" in response  # type: ignore
                assert len(response["data"]) == 9  # type: ignore  # 9 results in mock response
                assert "search_query" in response  # type: ignore

                # Validate first few results
                expected_results = [
                    {
                        "book_id": 0,
                        "text": "abababababa_0562efee-0f1f-4b6b-9ca3-1a160f124ad8",
                        "distance": 10.240219,
                    },
                    {
                        "book_id": 1,
                        "text": "abababababa_9a13e8f3-bb1e-487f-b555-b8ae4b127243",
                        "distance": 10.240219,
                    },
                    {
                        "book_id": 2,
                        "text": "abababababa_870f47f1-23ec-4364-ad30-6d364ba8ddb5",
                        "distance": 10.240219,
                    },
                ]

                for idx, expected in enumerate(expected_results):
                    result = response["data"][idx]  # type: ignore
                    assert "score" in result
                    assert result["score"] == expected["distance"]  # type: ignore
                    assert "content" in result
                    assert len(result["content"]) > 0  # type: ignore
                    assert result["content"][0]["type"] == "text"  # type: ignore
                    assert "text" in result["content"][0]  # type: ignore
                    assert result["content"][0]["text"] == expected["text"]  # type: ignore
                    assert "attributes" in result
                    assert result["attributes"]["book_id"] == expected["book_id"]  # type: ignore
                    assert "book_intro_text" not in result["attributes"]  # type: ignore  # Should be in content, not attributes

    def _extract_request_body(self, mock_post):
        call_args = mock_post.call_args
        request_data_str = call_args.kwargs.get("data")
        if request_data_str:
            return json.loads(request_data_str)
        request_data = call_args.kwargs.get("json")
        if request_data is None and len(call_args.args) > 0 and isinstance(call_args.args[0], dict):
            request_data = call_args.args[0]
        return request_data

    def test_user_supplied_db_and_partition_are_dropped(self):
        """User-supplied dbName / partitionNames must not be forwarded to Milvus."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_MILVUS_SEARCH_RESPONSE
        mock_response.text = json.dumps(MOCK_MILVUS_SEARCH_RESPONSE)

        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = MOCK_EMBEDDING_RESPONSE

            with patch(  # test-quality-ok: isolates tenant-field request transformation from network transport
                "litellm.llms.custom_httpx.http_handler.HTTPHandler.post"
            ) as mock_post:
                mock_post.return_value = mock_response

                vector_store_search(
                    query="what is machine learning?",
                    vector_store_id="book_2",
                    custom_llm_provider="milvus",
                    api_base="https://in03-test.serverless.aws-eu-central-1.cloud.zilliz.com",
                    api_key="mock_milvus_api_key",
                    litellm_embedding_model="text-embedding-3-large",
                    litellm_embedding_config={
                        "api_key": "mock_openai_api_key",
                    },
                    outputFields=["book_intro_text"],
                    annsField="book_intro_vector",
                    milvus_text_field="book_intro_text",
                    dbName="other_tenant_db",
                    partitionNames=["other_tenant_partition"],
                )

                mock_post.assert_called_once()
                request_data = self._extract_request_body(mock_post)
                assert request_data is not None
                assert "dbName" not in request_data
                assert "partitionNames" not in request_data
                assert request_data["collectionName"] == "book_2"
                assert request_data["annsField"] == "book_intro_vector"
                assert request_data["outputFields"] == ["book_intro_text"]

    def test_backend_configured_db_and_partition_are_forwarded(self):
        """milvus_db_name / milvus_partition_names from litellm_params must be sent."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_MILVUS_SEARCH_RESPONSE
        mock_response.text = json.dumps(MOCK_MILVUS_SEARCH_RESPONSE)

        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = MOCK_EMBEDDING_RESPONSE

            with patch(  # test-quality-ok: isolates persisted tenant configuration mapping from network transport
                "litellm.llms.custom_httpx.http_handler.HTTPHandler.post"
            ) as mock_post:
                mock_post.return_value = mock_response

                vector_store_search(
                    query="what is machine learning?",
                    vector_store_id="book_2",
                    custom_llm_provider="milvus",
                    api_base="https://in03-test.serverless.aws-eu-central-1.cloud.zilliz.com",
                    api_key="mock_milvus_api_key",
                    litellm_embedding_model="text-embedding-3-large",
                    litellm_embedding_config={
                        "api_key": "mock_openai_api_key",
                    },
                    outputFields=["book_intro_text"],
                    annsField="book_intro_vector",
                    milvus_text_field="book_intro_text",
                    milvus_db_name="tenant_a_db",
                    milvus_partition_names=["tenant_a_partition"],
                )

                mock_post.assert_called_once()
                request_data = self._extract_request_body(mock_post)
                assert request_data is not None
                assert request_data["dbName"] == "tenant_a_db"
                assert request_data["partitionNames"] == ["tenant_a_partition"]

    def test_user_params_cannot_override_backend_db_and_partition(self):
        """Backend-config dbName/partitionNames must win over user-supplied values."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_MILVUS_SEARCH_RESPONSE
        mock_response.text = json.dumps(MOCK_MILVUS_SEARCH_RESPONSE)

        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = MOCK_EMBEDDING_RESPONSE

            with patch(  # test-quality-ok: isolates precedence validation from network transport
                "litellm.llms.custom_httpx.http_handler.HTTPHandler.post"
            ) as mock_post:
                mock_post.return_value = mock_response

                vector_store_search(
                    query="what is machine learning?",
                    vector_store_id="book_2",
                    custom_llm_provider="milvus",
                    api_base="https://in03-test.serverless.aws-eu-central-1.cloud.zilliz.com",
                    api_key="mock_milvus_api_key",
                    litellm_embedding_model="text-embedding-3-large",
                    litellm_embedding_config={
                        "api_key": "mock_openai_api_key",
                    },
                    outputFields=["book_intro_text"],
                    annsField="book_intro_vector",
                    milvus_text_field="book_intro_text",
                    milvus_db_name="tenant_a_db",
                    milvus_partition_names=["tenant_a_partition"],
                    dbName="other_tenant_db",
                    partitionNames=["other_tenant_partition"],
                )

                mock_post.assert_called_once()
                request_data = self._extract_request_body(mock_post)
                assert request_data is not None
                assert request_data["dbName"] == "tenant_a_db"
                assert request_data["partitionNames"] == ["tenant_a_partition"]

    def test_grpc_search_uses_pymilvus_client(self):
        mock_client = MagicMock()
        mock_client.search.return_value = [[MockPyMilvusHit(id=7, distance=0.91, entity={})]]
        mock_embedding = MagicMock(return_value=MOCK_EMBEDDING_RESPONSE)
        config = MilvusGRPCVectorStoreConfig(sync_client=mock_client, embedding_fn=mock_embedding)
        response = config.execute_search_vector_store_request(
            query="what is machine learning?",
            vector_store_id="book_2",
            vector_store_search_optional_params=cast(
                VectorStoreSearchOptionalRequestParams,
                {
                    "outputFields": ["book_intro_text", "category"],
                    "annsField": "book_intro_vector",
                    "limit": 3,
                    "filter": 'category == "reference"',
                },
            ),
            litellm_logging_obj=MagicMock(),
            litellm_params={
                "api_base": "https://milvus.example.com:19530",
                "api_key": "mock_milvus_api_key",
                "litellm_embedding_model": "text-embedding-3-large",
                "litellm_embedding_config": {"api_key": "mock_openai_api_key"},
                "milvus_text_field": "book_intro_text",
                "milvus_db_name": "tenant_a_db",
                "milvus_partition_names": ["tenant_a_partition"],
            },
        )

        mock_embedding.assert_called_once_with(
            "text-embedding-3-large",
            "what is machine learning?",
            {"api_key": "mock_openai_api_key"},
        )
        mock_client.search.assert_called_once()
        search_kwargs = mock_client.search.call_args.kwargs
        assert search_kwargs["collection_name"] == "book_2"
        assert search_kwargs["anns_field"] == "book_intro_vector"
        assert search_kwargs["limit"] == 3
        assert search_kwargs["filter"] == 'category == "reference"'
        assert search_kwargs["output_fields"] == ["book_intro_text", "category"]
        assert search_kwargs["partition_names"] == ["tenant_a_partition"]
        assert response["search_query"] == "what is machine learning?"
        assert response["data"] == [
            {
                "score": 0.91,
                "content": [{"text": "closest result", "type": "text"}],
                "file_id": None,
                "filename": None,
                "attributes": {"category": "reference"},
            }
        ]

    @pytest.mark.asyncio
    async def test_async_grpc_search_infers_vector_field_and_requests_text_by_default(self):
        mock_client = MagicMock()
        mock_client.search = AsyncMock(
            return_value=[
                [
                    {
                        "id": 8,
                        "distance": 0.88,
                        "entity": {"book_intro_text": "async result"},
                    }
                ]
            ]
        )
        mock_client.close = AsyncMock()
        mock_embedding = AsyncMock(return_value=MOCK_EMBEDDING_RESPONSE)
        config = MilvusGRPCVectorStoreConfig(async_client=mock_client, aembedding_fn=mock_embedding)
        response = await config.aexecute_search_vector_store_request(
            query=["what is", "machine learning?"],
            vector_store_id="book_2",
            vector_store_search_optional_params=cast(
                VectorStoreSearchOptionalRequestParams,
                {
                    "max_num_results": 2,
                },
            ),
            litellm_logging_obj=MagicMock(),
            litellm_params={
                "api_base": "http://localhost:19530",
                "litellm_embedding_model": "text-embedding-3-large",
                "milvus_text_field": "book_intro_text",
            },
        )

        mock_embedding.assert_awaited_once_with(
            "text-embedding-3-large",
            "what is machine learning?",
            {},
        )
        assert mock_client.search.await_args.kwargs["limit"] == 2
        assert mock_client.search.await_args.kwargs["anns_field"] is None
        assert mock_client.search.await_args.kwargs["output_fields"] == ["book_intro_text"]
        assert response["data"][0]["content"][0]["text"] == "async result"

    def test_grpc_search_always_requests_configured_text_field(self):
        mock_client = MagicMock()
        mock_client.search.return_value = [
            [
                {
                    "id": 9,
                    "distance": 0.87,
                    "entity": {
                        "body": "result text",
                        "category": "reference",
                    },
                }
            ]
        ]
        config = MilvusGRPCVectorStoreConfig(
            sync_client=mock_client,
            embedding_fn=MagicMock(return_value=MOCK_EMBEDDING_RESPONSE),
        )

        response = config.execute_search_vector_store_request(
            query="what is machine learning?",
            vector_store_id="documents",
            vector_store_search_optional_params=cast(
                VectorStoreSearchOptionalRequestParams,
                {"outputFields": ["category"]},
            ),
            litellm_logging_obj=MagicMock(),
            litellm_params={
                "api_base": "http://localhost:19530",
                "litellm_embedding_model": "text-embedding-3-large",
                "milvus_text_field": "body",
            },
        )

        assert mock_client.search.call_args.kwargs["output_fields"] == ["category", "body"]
        assert response["data"][0]["content"][0]["text"] == "result text"

    @pytest.mark.parametrize(
        "optional_params",
        [
            {"limit": 0},
            {"limit": 51},
            {"max_num_results": 0},
            {"max_num_results": 51},
        ],
    )
    def test_grpc_search_rejects_invalid_result_limits(self, optional_params):
        mock_client = MagicMock()
        mock_embedding = MagicMock(return_value=MOCK_EMBEDDING_RESPONSE)
        config = MilvusGRPCVectorStoreConfig(sync_client=mock_client, embedding_fn=mock_embedding)

        with pytest.raises(ValueError, match=r"Input should be (greater|less) than or equal"):
            config.execute_search_vector_store_request(
                query="what is machine learning?",
                vector_store_id="book_2",
                vector_store_search_optional_params=optional_params,
                litellm_logging_obj=MagicMock(),
                litellm_params={
                    "api_base": "https://milvus.example.com:19530",
                    "litellm_embedding_model": "openai/text-embedding-3-small",
                },
            )

        mock_embedding.assert_not_called()
        mock_client.search.assert_not_called()

    @pytest.mark.parametrize(
        ("parameter", "value"),
        [
            ("filters", {"type": "eq", "key": "category", "value": "reference"}),
            ("ranking_options", {"score_threshold": 0.5}),
            ("rewrite_query", True),
        ],
    )
    def test_grpc_search_rejects_unsupported_openai_params(self, parameter, value):
        mock_client = MagicMock()
        mock_embedding = MagicMock(return_value=MOCK_EMBEDDING_RESPONSE)
        config = MilvusGRPCVectorStoreConfig(sync_client=mock_client, embedding_fn=mock_embedding)

        with pytest.raises(litellm.BadRequestError, match=f"does not support the {parameter} parameter") as exc_info:
            config.execute_search_vector_store_request(
                query="what is machine learning?",
                vector_store_id="book_2",
                vector_store_search_optional_params=cast(VectorStoreSearchOptionalRequestParams, {parameter: value}),
                litellm_logging_obj=MagicMock(),
                litellm_params={
                    "api_base": "https://milvus.example.com:19530",
                    "litellm_embedding_model": "openai/text-embedding-3-small",
                },
            )

        assert exc_info.value.status_code == 400
        mock_embedding.assert_not_called()
        mock_client.search.assert_not_called()

    def test_grpc_transport_selects_direct_config(self):
        config = ProviderConfigManager.get_provider_vector_stores_config(
            provider=litellm.LlmProviders.MILVUS,
            transport="grpc",
        )
        assert isinstance(config, MilvusGRPCVectorStoreConfig)

    def test_milvus_transport_defaults_to_rest(self):
        config = ProviderConfigManager.get_provider_vector_stores_config(
            provider=litellm.LlmProviders.MILVUS,
        )
        assert isinstance(config, MilvusVectorStoreConfig)

    def test_public_grpc_search_passes_connection_settings_to_pymilvus(self):
        mock_client = MagicMock()
        mock_client.search.return_value = [
            [
                {
                    "id": 9,
                    "distance": 1.0,
                    "entity": {"text": "secured result"},
                }
            ]
        ]

        def embedding_response(request: httpx.Request, *, stream: bool = False) -> httpx.Response:
            return httpx.Response(
                200,
                request=request,
                json={
                    "data": [
                        {
                            "embedding": [1.0, 0.0],
                            "index": 0,
                            "object": "embedding",
                        }
                    ],
                    "model": "test-embedding",
                    "object": "list",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )

        with (
            patch("httpx.Client.send", side_effect=embedding_response),
            patch("pymilvus.MilvusClient", return_value=mock_client) as client_class,
        ):
            response = vector_store_search(
                query="transport probe",
                vector_store_id="documents",
                custom_llm_provider="milvus",
                milvus_transport="grpc",
                api_base="https://milvus.example.com:19530",
                api_key="root:Milvus",
                litellm_embedding_model="openai/test-embedding",
                litellm_embedding_config={
                    "api_base": "https://embeddings.example/v1",
                    "api_key": "embedding-key",
                },
                milvus_db_name="tenant_db",
                annsField="vector",
                outputFields=["text"],
                milvus_text_field="text",
                timeout=17,
            )

        client_class.assert_called_once_with(
            uri="https://milvus.example.com:19530",
            token="root:Milvus",
            db_name="tenant_db",
            timeout=17.0,
            dedicated=True,
        )
        mock_client.close.assert_called_once_with()
        assert response["data"][0]["content"][0]["text"] == "secured result"

    @pytest.mark.asyncio
    async def test_async_grpc_uses_distinct_timeouts_and_releases_dedicated_client(self):
        mock_client = MagicMock()
        mock_client.search = AsyncMock(return_value=[[]])
        mock_client.close = AsyncMock()
        embedding_executor = MagicMock()
        embedding_executor.aembed = AsyncMock(return_value=MOCK_EMBEDDING_RESPONSE)
        timeout = httpx.Timeout(connect=3, read=11, write=13, pool=17)

        with patch("pymilvus.AsyncMilvusClient", return_value=mock_client) as client_class:
            response = await MilvusGRPCVectorStoreConfig().aexecute_search_vector_store_request(
                query="transport probe",
                vector_store_id="documents",
                vector_store_search_optional_params={},
                litellm_logging_obj=MagicMock(),
                litellm_params={
                    "api_base": "http://milvus.example.com:19530",
                    "api_key": "root:Milvus",
                    "litellm_embedding_model": "embedding-alias",
                },
                embedding_executor=embedding_executor,
                timeout=timeout,
            )

        client_class.assert_called_once_with(
            uri="http://milvus.example.com:19530",
            token="root:Milvus",
            db_name="",
            timeout=3,
            dedicated=True,
        )
        assert mock_client.search.await_args.kwargs["timeout"] == 11
        assert response["data"] == []
        embedding_executor.aembed.assert_awaited_once_with("embedding-alias", "transport probe", {})
        mock_client.close.assert_awaited_once_with()

    def test_http_and_https_targets_get_distinct_dedicated_clients(self):
        clients = [MagicMock(), MagicMock()]
        for client in clients:
            client.search.return_value = [[]]
        embedding_executor = MagicMock()
        embedding_executor.embed.return_value = MOCK_EMBEDDING_RESPONSE

        responses = []
        with patch("pymilvus.MilvusClient", side_effect=clients) as client_class:
            for uri in ("http://milvus.example.com:19530", "https://milvus.example.com:19530"):
                responses.append(
                    MilvusGRPCVectorStoreConfig().execute_search_vector_store_request(
                        query="transport probe",
                        vector_store_id="documents",
                        vector_store_search_optional_params={},
                        litellm_logging_obj=MagicMock(),
                        litellm_params={
                            "api_base": uri,
                            "litellm_embedding_model": "embedding-alias",
                        },
                        embedding_executor=embedding_executor,
                    )
                )

        assert [response["data"] for response in responses] == [[], []]
        assert [call.kwargs["uri"] for call in client_class.call_args_list] == [
            "http://milvus.example.com:19530",
            "https://milvus.example.com:19530",
        ]
        assert all(call.kwargs["dedicated"] is True for call in client_class.call_args_list)
        for client in clients:
            client.close.assert_called_once_with()

    def test_invalid_milvus_transport_is_rejected(self):
        with pytest.raises(ValueError, match="milvus_transport"):
            GenericLiteLLMParams.model_validate({"milvus_transport": "http"})


# @pytest.mark.parametrize("sync_mode", [True, False])
# @pytest.mark.asyncio
# async def test_basic_search_vector_store(sync_mode):
#     """Integration test with real Milvus API (requires credentials)"""
#     litellm._turn_on_debug()
#     litellm.set_verbose = True
#     base_request_args = {
#         "vector_store_id": "book_2",
#         "custom_llm_provider": "milvus",
#         "api_base": "https://in03-18505f064ffbc6f.serverless.aws-eu-central-1.cloud.zilliz.com",
#         "litellm_embedding_model": "text-embedding-3-large",
#         "litellm_embedding_config": {
#             "api_key": os.getenv("OPENAI_API_KEY"),
#         },
#         "default_output_fields": [
#             "book_intro_text"
#         ],  # field containing the text to return in the response
#         "default_anns_field": "book_intro_vector",
#     }
#     default_query = base_request_args.pop("query", "Basic ping")
#     print(f"base_request_args: {base_request_args}")
#     try:
#         if sync_mode:
#             response = vector_store_search(query=default_query, **base_request_args)
#         else:
#             response = await vector_store_asearch(
#                 query=default_query, **base_request_args
#             )
#     except litellm.InternalServerError:
#         pytest.skip("Skipping test due to litellm.InternalServerError")

#     print("litellm response=", json.dumps(response, indent=4, default=str))
#     assert len(response["data"]) > 0  # type: ignore


if __name__ == "__main__":
    # Run tests
    import asyncio

    test = TestMilvusVectorStore()

    print("Running async mock test...")
    asyncio.run(test.test_basic_search_with_mock_async())

    print("\nRunning sync mock test...")
    test.test_basic_search_with_mock_sync()

    print("\n✅ All mock tests passed!")
