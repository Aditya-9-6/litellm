"""
Tests for Milvus Vector Store
"""

import asyncio
import json
from contextlib import nullcontext
from functools import partial
from typing import Final, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
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


@pytest.fixture
def milvus_search_result() -> object:
    from pymilvus import DataType
    from pymilvus.client.search_result import SearchResult
    from pymilvus.grpc_gen import schema_pb2

    return SearchResult(
        schema_pb2.SearchResultData(
            num_queries=1,
            topks=[1],
            scores=[1.0],
            ids={"int_id": {"data": [7]}},
            output_fields=["book_intro_text", "category"],
            fields_data=[
                {
                    "field_name": "$meta",
                    "type": DataType.JSON,
                    "is_dynamic": True,
                    "scalars": {
                        "json_data": {"data": [b'{"book_intro_text":"closest result","category":"reference"}']}
                    },
                }
            ],
        )
    )


class TestMilvusVectorStore:
    """Test Milvus Vector Store with mocked responses"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", (False, True))
    @pytest.mark.parametrize("query", ("what is machine learning?", ["what is", "machine learning?"]))
    async def test_rest_search(self, async_mode: bool, query: str | list[str]) -> None:
        mock_post: Final = (AsyncMock if async_mode else MagicMock)(
            return_value=httpx.Response(200, json=MOCK_MILVUS_SEARCH_RESPONSE)
        )
        client: Final = MagicMock(spec=AsyncHTTPHandler if async_mode else HTTPHandler, post=mock_post)
        search: Final = partial(
            vector_store_asearch if async_mode else vector_store_search,
            query=query,
            vector_store_id="book_2",
            custom_llm_provider="milvus",
            api_base="https://milvus.example.com",
            api_key="mock_milvus_api_key",
            litellm_embedding_model="text-embedding-3-large",
            litellm_embedding_config={"api_key": "mock_openai_api_key"},
            outputFields=["book_intro_text"],
            annsField="book_intro_vector",
            milvus_text_field="book_intro_text",
            client=client,
        )
        with patch("litellm.embedding", return_value=MOCK_EMBEDDING_RESPONSE) as mock_embedding:
            response: Final = await search() if async_mode else search()

        mock_embedding.assert_called_once_with(
            model="text-embedding-3-large", input=["what is machine learning?"], api_key="mock_openai_api_key"
        )
        mock_post.assert_called_once()
        assert mock_post.call_args.kwargs["url"] == "https://milvus.example.com/v2/vectordb/entities/search"
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer mock_milvus_api_key"
        assert json.loads(mock_post.call_args.kwargs["data"]) == {
            "collectionName": "book_2",
            "data": [MOCK_EMBEDDING_RESPONSE.data[0]["embedding"]],
            "annsField": "book_intro_vector",
            "outputFields": ["book_intro_text"],
        }
        assert response == {
            "object": "vector_store.search_results.page",
            "search_query": "",
            "data": [
                {
                    "score": result["distance"],
                    "content": [{"text": result["book_intro_text"], "type": "text"}],
                    "file_id": None,
                    "filename": None,
                    "attributes": {"book_id": result["book_id"]},
                }
                for result in MOCK_MILVUS_SEARCH_RESPONSE["data"]
            ],
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", (False, True))
    async def test_grpc_rejects_empty_embedding_before_search(self, async_mode: bool) -> None:
        client: Final = MagicMock()
        embedding: Final = (AsyncMock if async_mode else MagicMock)(return_value={"data": []})
        config: Final = MilvusGRPCVectorStoreConfig(
            sync_client=client, async_client=client, embedding_fn=embedding, aembedding_fn=embedding
        )
        search: Final = partial(
            config.aexecute_search_vector_store_request if async_mode else config.execute_search_vector_store_request,
            vector_store_id="documents",
            query="transport probe",
            vector_store_search_optional_params={},
            litellm_logging_obj=MagicMock(),
            litellm_params={"litellm_embedding_model": "embedding-alias"},
        )
        with pytest.raises(ValueError, match="at least 1 item"):
            await search() if async_mode else search()
        client.search.assert_not_called()

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
                request_data = json.loads(mock_post.call_args.kwargs["data"])
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
                request_data = json.loads(mock_post.call_args.kwargs["data"])
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
                request_data = json.loads(mock_post.call_args.kwargs["data"])
                assert request_data is not None
                assert request_data["dbName"] == "tenant_a_db"
                assert request_data["partitionNames"] == ["tenant_a_partition"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", (False, True))
    @pytest.mark.parametrize("output_fields", (("*",), ("category",)))
    async def test_grpc_search_options_match_across_sync_and_async(
        self, async_mode: bool, output_fields: tuple[str, ...], milvus_search_result: object
    ) -> None:
        mock_search: Final = (AsyncMock if async_mode else MagicMock)(return_value=milvus_search_result)
        mock_client: Final = MagicMock(search=mock_search, close=(AsyncMock if async_mode else MagicMock)())
        mock_embedding: Final = (AsyncMock if async_mode else MagicMock)(return_value=MOCK_EMBEDDING_RESPONSE)
        config: Final = MilvusGRPCVectorStoreConfig(embedding_fn=mock_embedding, aembedding_fn=mock_embedding)
        search: Final = partial(
            config.aexecute_search_vector_store_request if async_mode else config.execute_search_vector_store_request,
            query=("what is", "machine learning?"),
            vector_store_id="book_2",
            vector_store_search_optional_params=cast(
                VectorStoreSearchOptionalRequestParams,
                {
                    "outputFields": output_fields,
                    "annsField": "book_intro_vector",
                    "limit": 3,
                    "max_num_results": 2,
                    "filter": 'category == "reference"',
                    "offset": 1,
                    "groupingField": "category",
                    "searchParams": {"metric_type": "COSINE", "params": {"nprobe": 8}},
                    "consistencyLevel": "Strong",
                },
            ),
            litellm_logging_obj=MagicMock(),
            litellm_params={
                "api_base": "https://milvus.example.com:19530/",
                "api_key": "mock_milvus_api_key",
                "litellm_embedding_model": "text-embedding-3-large",
                "litellm_embedding_config": {"api_key": "mock_openai_api_key"},
                "milvus_text_field": "book_intro_text",
                "milvus_db_name": "tenant_a_db",
                "milvus_partition_names": ["tenant_a_partition"],
            },
            timeout=httpx.Timeout(connect=3, read=11, write=13, pool=17),
        )
        with patch(
            "pymilvus.AsyncMilvusClient" if async_mode else "pymilvus.MilvusClient", return_value=mock_client
        ) as client_class:
            response: Final = await search() if async_mode else search()

        client_class.assert_called_once_with(
            uri="https://milvus.example.com:19530",
            token="mock_milvus_api_key",
            db_name="tenant_a_db",
            timeout=3,
            dedicated=True,
        )
        mock_embedding.assert_called_once_with(
            "text-embedding-3-large", "what is machine learning?", {"api_key": "mock_openai_api_key"}
        )
        mock_search.assert_called_once_with(
            collection_name="book_2",
            data=[MOCK_EMBEDDING_RESPONSE.data[0]["embedding"]],
            anns_field="book_intro_vector",
            limit=2,
            filter='category == "reference"',
            offset=1,
            group_by_field="category",
            output_fields=["*"] if output_fields == ("*",) else ["category", "book_intro_text"],
            search_params={"metric_type": "COSINE", "params": {"nprobe": 8}},
            consistency_level="Strong",
            partition_names=["tenant_a_partition"],
            timeout=11,
        )
        assert response == {
            "object": "vector_store.search_results.page",
            "search_query": "what is machine learning?",
            "data": [
                {
                    "score": 1.0,
                    "content": [{"text": "closest result", "type": "text"}],
                    "file_id": None,
                    "filename": None,
                    "attributes": {"category": "reference"},
                }
            ],
        }

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

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", (False, True))
    async def test_public_grpc_search_passes_connection_settings_to_pymilvus(self, async_mode: bool) -> None:
        mock_client = MagicMock()
        mock_client.search = (AsyncMock if async_mode else MagicMock)(
            return_value=[
                [
                    {
                        "id": 9,
                        "distance": 1.0,
                        "entity": {"text": "secured result"},
                    }
                ]
            ]
        )
        mock_client.close = (AsyncMock if async_mode else MagicMock)()

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

        transport: Final = httpx.MockTransport(embedding_response)
        async with httpx.AsyncClient(transport=transport) as async_http_client:
            with (
                httpx.Client(transport=transport) as http_client,
                patch(
                    "pymilvus.AsyncMilvusClient" if async_mode else "pymilvus.MilvusClient", return_value=mock_client
                ) as client_class,
            ):
                search: Final = partial(
                    vector_store_asearch if async_mode else vector_store_search,
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
                        "client": AsyncOpenAI(api_key="embedding-key", http_client=async_http_client)
                        if async_mode
                        else OpenAI(api_key="embedding-key", http_client=http_client),
                    },
                    milvus_db_name="tenant_db",
                    annsField="vector",
                    outputFields=["text"],
                    milvus_text_field="text",
                    timeout=17,
                )

                response: Final = await search() if async_mode else search()

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
    @pytest.mark.parametrize("injected", (False, True))
    @pytest.mark.parametrize(
        ("async_mode", "result", "error"),
        (
            (False, [[]], None),
            (True, [[]], None),
            (False, [[]], RuntimeError),
            (True, [[]], RuntimeError),
            (True, [[]], asyncio.CancelledError),
            (False, [[None]], TypeError),
            (True, [[None]], TypeError),
            (False, [[{1: "invalid key"}]], TypeError),
            (True, [[{1: "invalid key"}]], TypeError),
        ),
    )
    async def test_grpc_client_ownership_on_success_error_and_cancellation(
        self, injected: bool, async_mode: bool, result: object, error: type[BaseException] | None
    ) -> None:
        mock_client: Final = MagicMock(
            search=(AsyncMock if async_mode else MagicMock)(
                return_value=result, side_effect=None if error is TypeError else error
            ),
            close=(AsyncMock if async_mode else MagicMock)(),
        )
        embedding_executor: Final = MagicMock(
            embed=MagicMock(return_value=MOCK_EMBEDDING_RESPONSE),
            aembed=AsyncMock(return_value=MOCK_EMBEDDING_RESPONSE),
        )
        config: Final = MilvusGRPCVectorStoreConfig(
            sync_client=mock_client if injected and not async_mode else None,
            async_client=mock_client if injected and async_mode else None,
        )
        search: Final = partial(
            config.aexecute_search_vector_store_request if async_mode else config.execute_search_vector_store_request,
            query="transport probe",
            vector_store_id="documents",
            vector_store_search_optional_params={},
            litellm_logging_obj=MagicMock(),
            litellm_params={
                "api_base": "http://milvus.example.com:19530",
                "litellm_embedding_model": "embedding-alias",
            },
            embedding_executor=embedding_executor,
        )
        with (
            patch(
                "pymilvus.AsyncMilvusClient" if async_mode else "pymilvus.MilvusClient", return_value=mock_client
            ) as client_class,
            pytest.raises(error) if error else nullcontext(),
        ):
            response: Final = await search() if async_mode else search()
            assert response["data"] == []

        assert client_class.call_count == (0 if injected else 1)
        if injected:
            mock_client.close.assert_not_called()
        elif async_mode:
            mock_client.close.assert_awaited_once_with()
        else:
            mock_client.close.assert_called_once_with()
        (embedding_executor.aembed if async_mode else embedding_executor.embed).assert_called_once_with(
            "embedding-alias", "transport probe", {}
        )

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
