from .types import (
    RerankerModelCreateParams,
    RerankerModelResponseParams,
)
from ..errors import GraxonAPIError, GraxonNetworkError
from typing import Any, Dict
import httpx
import uuid


class RerankerModel:
    """Client for managing Reranker Models in the Graxon API.

    Examples:
        ```python
        from graxon.client import GraxonAsyncClient
        from graxon.reranker_models.types import (
            RerankerModelCreateParams,
            RerankerModelProvider,
            RerankerModelProviderType,
        )

        client = GraxonAsyncClient(api_key="graxon_api_key", base_url="http://localhost:8888")
        model = await client.reranker_models.create(
            org_id="test",
            request=RerankerModelCreateParams(
                org_id="test",
                name="Jina Reranker Test Model",
                provider_type=RerankerModelProviderType.CLOUD,
                provider=RerankerModelProvider.JINA,
                model_name="Jina Reranker v2",
                model_id="jina-reranker-v2",
                description="Jina Reranker Test Model",
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

        self._reranker_model_prefix = "/api/rerankers"

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
        request: RerankerModelCreateParams,
    ) -> RerankerModelResponseParams:
        """Creates a new Reranker model for an organization.

        Args:
            org_id: The unique identifier of the organization to create the
                Reranker model under.
            request: The parameters for the Reranker model to create, including its
                `name`, `model_name`, `model_id`, `provider`, `provider_type`, and `size_in_gb`.

        Returns:
            RerankerModelResponseParams: The newly created Reranker model, including its
            assigned `id`.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            create_response = await client.reranker_models.create(
                org_id="test",
                request=RerankerModelCreateParams(
                    org_id="test",
                    name="Jina Reranker Test Model",
                    provider_type=RerankerModelProviderType.CLOUD,
                    provider=RerankerModelProvider.JINA,
                    model_name="Jina Reranker v2",
                    model_id="jina-reranker-v2",
                    description="Jina Reranker Test Model",
                    size_in_gb=0.0,
                ),
            )
            print(create_response.id)
            ```
        """
        payload = request.model_dump()

        res_data = await self._request(
            "POST",
            f"{self._reranker_model_prefix}/{org_id}/create",
            json=payload,
        )

        data = res_data.get("data")

        if not data:
            raise GraxonAPIError(
                "Graxon API Error: Response missing 'data' payload"
            )

        return RerankerModelResponseParams(**data)

    async def create_multiple(
        self,
        org_id: str,
        request: list[RerankerModelCreateParams],
    ) -> dict[str, Any]:
        """Creates multiple Reranker models for an organization in a single request.

        Args:
            org_id: The unique identifier of the organization to create the
                Reranker models under.
            request: A list of parameters for the Reranker models to create.

        Returns:
            dict[str, Any]: A dictionary containing the response payload with details
            of the created models.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            multiple_create_response = await client.reranker_models.create_multiple(
                org_id="test",
                request=[
                    RerankerModelCreateParams(
                        org_id="test",
                        name="Cohere Reranker Test Model",
                        provider_type=RerankerModelProviderType.CLOUD,
                        provider=RerankerModelProvider.COHERE,
                        model_name="Cohere Rerank",
                        model_id="rerank-v3.5",
                        description="Cohere Reranker Test Model",
                        size_in_gb=0.0,
                    ),
                    RerankerModelCreateParams(
                        org_id="test",
                        name="Xenova Reranker Test Model",
                        provider_type=RerankerModelProviderType.LOCAL,
                        provider=RerankerModelProvider.XENOVA,
                        model_name="Xenova Reranker",
                        model_id="Xenova/ms-marco-MiniLM-L-6-v2",
                        description="Xenova Local Reranker Test Model",
                        size_in_gb=0.1,
                    ),
                ],
            )
            print(multiple_create_response)
            ```
        """
        payload = [item.model_dump() for item in request]

        res_data = await self._request(
            "POST",
            f"{self._reranker_model_prefix}/{org_id}/create-multiple",
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
        reranker_model_id: uuid.UUID,
    ) -> RerankerModelResponseParams:
        """Retrieves a specific Reranker model by its ID.

        Args:
            org_id: The unique identifier of the organization.
            reranker_model_id: The UUID of the Reranker model to retrieve.

        Returns:
            RerankerModelResponseParams: The requested Reranker model details.

        Raises:
            GraxonAPIError: If the model is not found, or the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            get_response = await client.reranker_models.get(
                org_id="test",
                reranker_model_id=create_response.id,
            )
            print(get_response.name)
            ```
        """
        res_data = await self._request(
            "GET",
            f"{self._reranker_model_prefix}/{org_id}/get/{reranker_model_id}",
        )

        data = res_data.get("data")

        if not data:
            raise GraxonAPIError(
                "Graxon API Error: Response missing 'data' payload"
            )

        return RerankerModelResponseParams(**data)

    async def list(
        self,
        org_id: str,
    ) -> list[RerankerModelResponseParams]:
        """Retrieves a list of all Reranker models for a specific organization.

        Args:
            org_id: The unique identifier of the organization.

        Returns:
            list[RerankerModelResponseParams]: A list of all Reranker models belonging to the organization.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            list_response = await client.reranker_models.list(
                org_id="test",
            )
            for model in list_response:
                print(model.name)
            ```
        """
        res_data = await self._request(
            "GET",
            f"{self._reranker_model_prefix}/{org_id}/get/all",
        )

        # Handle nested {"data": {"data": [...]}} response
        wrapper_data = res_data.get("data") or {}
        list_data = wrapper_data.get("data") or []

        return [
            RerankerModelResponseParams(**item)
            for item in list_data
        ]

    async def delete(
        self,
        org_id: str,
        reranker_model_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Deletes a specific Reranker model by its ID.

        Args:
            org_id: The unique identifier of the organization.
            reranker_model_id: The UUID of the Reranker model to delete.

        Returns:
            dict[str, Any]: A dictionary containing the response payload confirming deletion.

        Raises:
            GraxonAPIError: If the model is not found, or the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            delete_response = await client.reranker_models.delete(
                org_id="test",
                reranker_model_id=create_response.id,
            )
            print(delete_response)
            ```
        """
        res_data = await self._request(
            "DELETE",
            f"{self._reranker_model_prefix}/{org_id}/delete/{reranker_model_id}",
        )

        data = res_data.get("data")

        if not data:
            raise GraxonAPIError(
                "Graxon API Error: Response missing 'data' payload"
            )

        return data
