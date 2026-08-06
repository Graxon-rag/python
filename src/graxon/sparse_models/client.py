from .types import (
    SparseModelCreateParams,
    SparseModelResponseParams,
)
from ..errors import GraxonAPIError, GraxonNetworkError
from typing import Any, Dict
import httpx
import uuid


class SparseModel:
    """Client for managing Sparse Text Models in the Graxon API.

    Examples:
        ```python
        from graxon.client import GraxonAsyncClient
        from graxon.sparse_models.types import (
            SparseModelCreateParams,
            SparseModelProvider,
            SparseModelProviderType,
        )

        client = GraxonAsyncClient(api_key="graxon_api_key", base_url="http://localhost:8888")
        model = await client.sparse_models.create(
            org_id="test",
            request=SparseModelCreateParams(
                org_id="test",
                name="Pinecone Sparse Test Model",
                provider_type=SparseModelProviderType.CLOUD,
                provider=SparseModelProvider.PINECONE,
                model_name="Pinecone Sparse English",
                model_id="pinecone-sparse-english-v0",
                description="Pinecone Sparse Text Test Model",
                size_in_gb=0.0,
            ),
        )
        ```
    """

    def __init__(
        self,
        api_key: str | None,
        base_url: str = "http://localhost:8888",
        timeout: float | None = 120.0,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout

        headers = {
            "User-Agent": "graxon-python",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self._api_key:
            headers["GRAXON-API-KEY"] = self._api_key

        self._sparse_model_prefix = "/api/sparse-text-models"

        self._http_client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout),
        )

    async def close(self):
        """Closes the underlying HTTP connections."""
        await self._http_client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Internal helper to dispatch requests and handle standard API errors."""
        try:
            response = await self._http_client.request(
                method,
                path,
                **kwargs,
            )

            response.raise_for_status()

            res_data = response.json()
            success = res_data.get("success")

            if not success:
                error_msg = res_data.get(
                    "message",
                    "API returned success=false",
                )
                raise GraxonAPIError(
                    f"Graxon API Error: {error_msg}"
                )

            return res_data

        except httpx.HTTPStatusError as e:
            raise GraxonAPIError(
                f"Graxon API Error {e.response.status_code}: {e.response.text}"
            ) from None

        except httpx.RequestError as e:
            raise GraxonNetworkError(
                f"Failed to communicate with Graxon API: {str(e)}"
            ) from None

    async def create(
        self,
        org_id: str,
        request: SparseModelCreateParams,
    ) -> SparseModelResponseParams:
        """Creates a new Sparse Text model for an organization.

        Args:
            org_id: The unique identifier of the organization to create the
                Sparse model under.
            request: The parameters for the Sparse model to create, including its
                `name`, `model_name`, `model_id`, `provider`, `provider_type`, and `size_in_gb`.

        Returns:
            SparseModelResponseParams: The newly created Sparse model, including its
            assigned `id`.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            create_response = await client.sparse_models.create(
                org_id="test",
                request=SparseModelCreateParams(
                    org_id="test",
                    name="Pinecone Sparse Test Model",
                    provider_type=SparseModelProviderType.CLOUD,
                    provider=SparseModelProvider.PINECONE,
                    model_name="Pinecone Sparse English",
                    model_id="pinecone-sparse-english-v0",
                    description="Pinecone Sparse Text Test Model",
                    size_in_gb=0.0,
                ),
            )
            print(create_response.id)
            ```
        """
        payload = request.model_dump()

        res_data = await self._request(
            "POST",
            f"{self._sparse_model_prefix}/{org_id}/create",
            json=payload,
        )

        data = res_data.get("data")

        if not data:
            raise GraxonAPIError(
                "Graxon API Error: Response missing 'data' payload"
            )

        return SparseModelResponseParams(**data)

    async def create_multiple(
        self,
        org_id: str,
        request: list[SparseModelCreateParams],
    ) -> dict[str, Any]:
        """Creates multiple Sparse Text models for an organization in a single request.

        Args:
            org_id: The unique identifier of the organization to create the
                Sparse models under.
            request: A list of parameters for the Sparse models to create.

        Returns:
            dict[str, Any]: A dictionary containing the response payload with details
            of the created models.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            multiple_create_response = await client.sparse_models.create_multiple(
                org_id="test",
                request=[
                    SparseModelCreateParams(
                        org_id="test",
                        name="Qdrant Sparse Test Model",
                        provider_type=SparseModelProviderType.CLOUD,
                        provider=SparseModelProvider.QDRANT,
                        model_name="Qdrant Sparse",
                        model_id="qdrant-sparse",
                        description="Qdrant Sparse Text Test Model",
                        size_in_gb=0.0,
                    ),
                    SparseModelCreateParams(
                        org_id="test",
                        name="Prithvida Sparse Test Model",
                        provider_type=SparseModelProviderType.LOCAL,
                        provider=SparseModelProvider.PRITHVIDA,
                        model_name="Prithvida Sparse",
                        model_id="prithvida-sparse",
                        description="Prithvida Local Sparse Text Test Model",
                        size_in_gb=0.5,
                    ),
                ],
            )
            print(multiple_create_response)
            ```
        """
        payload = [item.model_dump() for item in request]

        res_data = await self._request(
            "POST",
            f"{self._sparse_model_prefix}/{org_id}/create-multiple",
            json=payload,
        )

        data = res_data.get("data")

        if not data:
            raise GraxonAPIError(
                "Graxon API Error: Response missing 'data' payload"
            )

        return data

    async def get(
        self,
        org_id: str,
        sparse_model_id: uuid.UUID,
    ) -> SparseModelResponseParams:
        """Retrieves a specific Sparse Text model by its ID.

        Args:
            org_id: The unique identifier of the organization.
            sparse_model_id: The UUID of the Sparse model to retrieve.

        Returns:
            SparseModelResponseParams: The requested Sparse model details.

        Raises:
            GraxonAPIError: If the model is not found, or the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            get_response = await client.sparse_models.get(
                org_id="test",
                sparse_model_id=create_response.id,
            )
            print(get_response.name)
            ```
        """
        res_data = await self._request(
            "GET",
            f"{self._sparse_model_prefix}/{org_id}/get/{sparse_model_id}",
        )

        data = res_data.get("data")

        if not data:
            raise GraxonAPIError(
                "Graxon API Error: Response missing 'data' payload"
            )

        return SparseModelResponseParams(**data)

    async def list(
        self,
        org_id: str,
    ) -> list[SparseModelResponseParams]:
        """Retrieves a list of all Sparse Text models for a specific organization.

        Args:
            org_id: The unique identifier of the organization.

        Returns:
            list[SparseModelResponseParams]: A list of all Sparse models belonging to the organization.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            list_response = await client.sparse_models.list(
                org_id="test",
            )
            for model in list_response:
                print(model.name)
            ```
        """
        res_data = await self._request(
            "GET",
            f"{self._sparse_model_prefix}/{org_id}/get/all",
        )

        # Handle nested {"data": {"data": [...]}} response
        wrapper_data = res_data.get("data") or {}
        list_data = wrapper_data.get("data") or []

        return [
            SparseModelResponseParams(**item)
            for item in list_data
        ]

    async def delete(self, org_id: str, sparse_model_id: uuid.UUID,) -> dict[str, Any]:
        """Deletes a specific Sparse Text model by its ID.

        Args:
            org_id: The unique identifier of the organization.
            sparse_model_id: The UUID of the Sparse model to delete.

        Returns:
            dict[str, Any]: A dictionary containing the response payload confirming deletion.

        Raises:
            GraxonAPIError: If the model is not found, or the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            delete_response = await client.sparse_models.delete(
                org_id="test",
                sparse_model_id=create_response.id,
            )
            print(delete_response)
            ```
        """
        res_data = await self._request(
            "DELETE",
            f"{self._sparse_model_prefix}/{org_id}/delete/{sparse_model_id}",
        )
        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")

        return data
