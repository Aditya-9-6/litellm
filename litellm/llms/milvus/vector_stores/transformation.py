from typing import TYPE_CHECKING, Any, Final

import httpx

import litellm
from litellm.llms.base_llm.vector_store.transformation import BaseVectorStoreConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_stores import (
    BaseVectorStoreAuthCredentials,
    VectorStoreCreateOptionalRequestParams,
    VectorStoreCreateResponse,
    VectorStoreIndexEndpoints,
    VectorStoreResultContent,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

MILVUS_OPTIONAL_PARAMS: Final = {
    "annsField",
    "limit",
    "filter",
    "offset",
    "groupingField",
    "outputFields",
    "searchParams",
    "consistencyLevel",
}


class MilvusVectorStoreConfig(BaseVectorStoreConfig):
    def validate_environment(self, headers: dict, litellm_params: GenericLiteLLMParams | None) -> dict:
        api_key: str | None = None
        if litellm_params is not None:
            api_key = litellm_params.api_key or get_secret_str("MILVUS_API_KEY")

        if not api_key:
            raise ValueError(
                "MILVUS_API_KEY is not set. Either set it in the litellm_params or set the MILVUS_API_KEY environment variable."
            )

        headers.update({"Authorization": f"Bearer {api_key}"})

        return headers

    def get_auth_credentials(self, litellm_params: dict) -> BaseVectorStoreAuthCredentials:
        api_key: Final = litellm_params.get("api_key")
        if not api_key:
            raise ValueError(
                "MILVUS_API_KEY is not set. Either set it in the litellm_params or set the MILVUS_API_KEY environment variable."
            )
        return {
            "headers": {
                "Authorization": f"Bearer {api_key}",
            },
        }

    def get_vector_store_endpoints_by_type(self) -> VectorStoreIndexEndpoints:
        return {
            "read": [
                ("POST", "/v2/vectordb/entities/search"),
                ("POST", "/v2/vectordb/entities/get"),
                ("POST", "/v2/vectordb/entities/query"),
            ],
            "write": [
                ("POST", "/v2/vectordb/entities/upsert"),
                ("POST", "/v2/vectordb/entities/insert"),
            ],
        }

    def map_openai_params(self, non_default_params: dict, optional_params: dict, drop_params: bool) -> dict:
        for param, value in non_default_params.items():
            if param in MILVUS_OPTIONAL_PARAMS:
                optional_params[param] = value
        return optional_params

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        resolved_api_base: Final = api_base or get_secret_str("MILVUS_API_BASE")
        if not resolved_api_base:
            raise ValueError(
                "Milvus API base URL is required. Set MILVUS_API_BASE environment variable or pass api_base in litellm_params."
            )

        return resolved_api_base.rstrip("/")

    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | list[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        query_text: Final = " ".join(query) if isinstance(query, list) else query
        embedding_model: Final = litellm_params.get("litellm_embedding_model")
        if not embedding_model:
            raise ValueError(
                "embedding_model is required in litellm_params for Milvus. You can call any litellm embedding model."
                "Example: litellm_params['embedding_model'] = 'azure/text-embedding-3-large'"
            )

        embedding_config: Final = litellm_params.get("litellm_embedding_config", {})
        if not embedding_config:
            raise ValueError(
                "embedding_config is required in litellm_params for Milvus. You can call any litellm embedding model."
                "Example: litellm_params['embedding_config'] = {'api_base': 'https://krris-mh44uf7y-eastus2.cognitiveservices.azure.com/', 'api_key': 'os.environ/AZURE_API_KEY', 'api_version': '2025-09-01'}"
            )

        try:
            embedding_response: Final = litellm.embedding(
                model=embedding_model,
                input=[query_text],
                **embedding_config,
            )
            query_vector: Final = embedding_response.data[0]["embedding"]
        except Exception as e:
            raise Exception(f"Failed to generate embedding for query: {e}")

        request_body: Final[dict[str, Any]] = {
            "collectionName": vector_store_id,
            "data": [query_vector],
            "annsField": "book_intro_vector",
            **vector_store_search_optional_params,
            **{
                field: value
                for field, value in (
                    ("dbName", litellm_params.get("milvus_db_name")),
                    ("partitionNames", litellm_params.get("milvus_partition_names")),
                )
                if value
            },
        }
        litellm_logging_obj.model_call_details["input"] = query_text
        litellm_logging_obj.model_call_details["embedding_model"] = embedding_model

        return f"{api_base}/v2/vectordb/entities/search", request_body

    def transform_search_vector_store_response(
        self, response: httpx.Response, litellm_logging_obj: LiteLLMLoggingObj
    ) -> VectorStoreSearchResponse:
        try:
            details: Final = litellm_logging_obj.model_call_details
            text_field: Final = details.get("optional_params", {}).get("milvus_text_field", "") or details.get(
                "litellm_params", {}
            ).get("milvus_text_field", "")
            return VectorStoreSearchResponse(
                object="vector_store.search_results.page",
                search_query=details.get("input", ""),
                data=[  # mutable-ok: VectorStoreSearchResponse requires list data
                    VectorStoreSearchResult(
                        score=result.get("distance", 0.0),
                        content=[  # mutable-ok: VectorStoreSearchResult requires list content
                            VectorStoreResultContent(text=result.get(text_field, ""), type="text")
                        ],
                        file_id=None,
                        filename=None,
                        attributes={  # mutable-ok: VectorStoreSearchResult requires dict attributes
                            key: value
                            for key, value in result.items()
                            if key not in ("id", "content", "distance", text_field)
                        },
                    )
                    for result in response.json().get("data", [])
                ],
            )

        except Exception as e:
            raise self.get_error_class(
                error_message=str(e),
                status_code=response.status_code,
                headers=response.headers,
            )

    def transform_create_vector_store_request(
        self,
        vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams,
        api_base: str,
    ) -> tuple[str, dict]:
        raise NotImplementedError

    def transform_create_vector_store_response(self, response: httpx.Response) -> VectorStoreCreateResponse:
        raise NotImplementedError
