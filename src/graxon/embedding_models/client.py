from .types import EmbeddingModelCreateParams, EmbeddingModelResponseParams, EmbeddingModelProvider
from ..errors import GraxonAPIError, GraxonNetworkError
from typing import Any, Dict
import httpx
import uuid


class EmbeddingModel:
    """Client for managing Embedding Models in the Graxon API.

    Examples:
        ```python
        from graxon.client import GraxonAsyncClient

        client = GraxonAsyncClient(api_key="graxon_api_key", base_url="http://localhost:8888")
        model = await client.embedding_models.create(
            org_id="test",
            request=EmbeddingModelCreateParams(
                org_id="test",
                name="OpenAI Test Model",
                model_name="text-embedding-ada-002",
                model_id="text-embedding-ada-002",
                provider=EmbeddingModelProvider.OPENAI,
                dimension=1536,
                description="OpenAI Test Model",
            ),
        )
        ```
    """

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
        """Closes the underlying HTTP connections.

        Examples:
            ```python
            await client.embedding_models.close()
            ```
        """
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
        """Creates a new embedding model for an organization.

        Args:
            org_id: The unique identifier of the organization to create the
                embedding model under.
            request: The parameters for the embedding model to create, including
                its `name`, `model_name`, `model_id`, `provider`, `dimension`,
                and `description`.

        Returns:
            EmbeddingModelResponseParams: The newly created embedding model,
            including its assigned `id`.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            model = await client.embedding_models.create(
                org_id="test",
                request=EmbeddingModelCreateParams(
                    org_id="test",
                    name="OpenAI Test Model",
                    model_name="text-embedding-ada-002",
                    model_id="text-embedding-ada-002",
                    provider=EmbeddingModelProvider.OPENAI,
                    dimension=1536,
                    description="OpenAI Test Model",
                ),
            )
            print(model.id)
            ```
        """
        payload = request.model_dump()
        res_data = await self._request("POST", f"{self._embedding_model_prefix}/{org_id}/create", json=payload)
        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return EmbeddingModelResponseParams(**data)

    async def create_multiple(self, org_id: str, request: list[EmbeddingModelCreateParams]) -> dict[str, Any]:
        """Creates multiple embedding models for an organization in a single request.

        Args:
            org_id: The unique identifier of the organization to create the
                embedding models under.
            request: A list of parameters, one per embedding model to create.

        Returns:
            dict[str, Any]: The raw API response payload describing the created
            embedding models.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            result = await client.embedding_models.create_multiple(
                org_id="test",
                request=[
                    EmbeddingModelCreateParams(
                        org_id="test",
                        name="GEMINI Test Model",
                        model_name="text-embedding-001 Model",
                        model_id="text-embedding-001",
                        provider=EmbeddingModelProvider.GEMINI,
                        dimension=1536,
                        description="GEMINI Test Model",
                    ),
                    EmbeddingModelCreateParams(
                        org_id="test",
                        name="VOYAGE Test Model",
                        model_name="embedding VOYAGE",
                        model_id="voyage-4-large",
                        provider=EmbeddingModelProvider.VOYAGE,
                        dimension=1536,
                        description="VOYAGE Test Model",
                    ),
                ],
            )
            print(result)
            ```
        """
        payload = [item.model_dump() for item in request]
        res_data = await self._request("POST", f"{self._embedding_model_prefix}/{org_id}/create-multiple", json=payload)

        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")

        return data

    async def get(self, org_id: str, embedding_model_id: uuid.UUID) -> EmbeddingModelResponseParams:
        """Retrieves a specific embedding model by ID.

        Args:
            org_id: The unique identifier of the organization the embedding
                model belongs to.
            embedding_model_id: The unique identifier of the embedding model to fetch.

        Returns:
            EmbeddingModelResponseParams: The matching embedding model.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            model = await client.embedding_models.get(org_id="test", embedding_model_id=create_response.id)
            print(model.name)
            ```
        """
        res_data = await self._request("GET", f"{self._embedding_model_prefix}/{org_id}/get/{embedding_model_id}")
        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return EmbeddingModelResponseParams(**data)

    async def list_by_provider(self, org_id: str, provider: EmbeddingModelProvider) -> list[EmbeddingModelResponseParams]:
        """Lists all embedding models for an organization filtered by provider.

        Args:
            org_id: The unique identifier of the organization to list embedding
                models for.
            provider: The provider to filter by (e.g. `EmbeddingModelProvider.OPENAI`).

        Returns:
            list[EmbeddingModelResponseParams]: All embedding models matching
            the given provider.

        Raises:
            GraxonAPIError: If the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            models = await client.embedding_models.list_by_provider(
                org_id="test", provider=EmbeddingModelProvider.OPENAI
            )
            for model in models:
                print(model.id, model.name)
            ```
        """
        res_data = await self._request("GET", f"{self._embedding_model_prefix}/{org_id}/get/all/provider/{provider.value}")
        # Handle the nested {"data": {"data": [...]}} structure
        wrapper_data = res_data.get("data") or {}
        list_data = wrapper_data.get("data") or []

        return [EmbeddingModelResponseParams(**item) for item in list_data]

    async def delete(self, org_id: str, embedding_model_id: uuid.UUID) -> dict[str, Any]:
        """Deletes an embedding model by ID.

        Args:
            org_id: The unique identifier of the organization the embedding
                model belongs to.
            embedding_model_id: The unique identifier of the embedding model to delete.

        Returns:
            dict[str, Any]: The raw API response confirming deletion.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            result = await client.embedding_models.delete(org_id="test", embedding_model_id=create_response.id)
            print(result)
            ```
        """
        res_data = await self._request("DELETE", f"{self._embedding_model_prefix}/{org_id}/delete/{embedding_model_id}")
        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return data
