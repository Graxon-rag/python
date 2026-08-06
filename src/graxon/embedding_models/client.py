from .types import EmbeddingModelCreateParams, EmbeddingModelResponseParams, EmbeddingModelProvider
from ..errors import GraxonAPIError, GraxonNetworkError
from typing import Any, Dict
import httpx
import uuid


class EmbeddingModel:
    def __init__(self, api_key: str | None, base_url: str = "http://localhost:8888", timeout: float | None = 120.0):
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout

        headers = {
            "User-Agent": "graxon-python",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key:
            headers["GRAXON-API-KEY"] = f"{self._api_key}"

        self._embedding_model_prefix = "/api/embedding-models"

        self._http_client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout)
        )

    async def close(self):
        """Closes the underlying HTTP connections."""
        await self._http_client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        """Internal helper to dispatch requests and handle standard API errors."""
        try:
            response = await self._http_client.request(method, path, **kwargs)
            response.raise_for_status()

            res_data = response.json()
            success = res_data.get("success")

            if not success:
                error_msg = res_data.get("message", "API returned success=false")
                raise GraxonAPIError(f"Graxon API Error: {error_msg}")

            return res_data

        except httpx.HTTPStatusError as e:
            raise GraxonAPIError(
                f"Graxon API Error {e.response.status_code}: {e.response.text}"
            ) from None

        except httpx.RequestError as e:
            raise GraxonNetworkError(
                f"Failed to communicate with Graxon API: {str(e)}"
            ) from None

    async def create(self, org_id: str, request: EmbeddingModelCreateParams) -> EmbeddingModelResponseParams:
        payload = request.model_dump()
        res_data = await self._request("POST", f"{self._embedding_model_prefix}/{org_id}/create", json=payload)
        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return EmbeddingModelResponseParams(**data)

    async def create_multiple(self, org_id: str, request: list[EmbeddingModelCreateParams]) -> dict[str, Any]:
        payload = [item.model_dump() for item in request]
        res_data = await self._request("POST", f"{self._embedding_model_prefix}/{org_id}/create-multiple", json=payload)

        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")

        return data

    async def get(self, org_id: str, embedding_model_id: uuid.UUID) -> EmbeddingModelResponseParams:
        res_data = await self._request("GET", f"{self._embedding_model_prefix}/{org_id}/get/{embedding_model_id}")
        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return EmbeddingModelResponseParams(**data)

    async def list_by_provider(self, org_id: str, provider: EmbeddingModelProvider) -> list[EmbeddingModelResponseParams]:
        res_data = await self._request("GET", f"{self._embedding_model_prefix}/{org_id}/get/all/provider/{provider.value}")
        # Handle the nested {"data": {"data": [...]}} structure
        wrapper_data = res_data.get("data") or {}
        list_data = wrapper_data.get("data") or []

        return [EmbeddingModelResponseParams(**item) for item in list_data]

    async def delete(self, org_id: str, embedding_model_id: uuid.UUID) -> dict[str, Any]:
        res_data = await self._request("DELETE", f"{self._embedding_model_prefix}/{org_id}/delete/{embedding_model_id}")
        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return data
