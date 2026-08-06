from .types import ModelCredentialCreateParams, ModelCredentialResponseParams
from ..errors import GraxonAPIError, GraxonNetworkError
from ..types import ModelProvider
from typing import Any, Dict
import httpx
import uuid


class ModelCredential:
    """Client for managing Model Credentials in the Graxon API.

    Examples:
        ```python
        from graxon.client import GraxonAsyncClient

        client = GraxonAsyncClient(api_key="graxon_api_key", base_url="http://localhost:8888")
        credential = await client.model_credentials.create(
            org_id="test",
            request=ModelCredentialCreateParams(
                org_id="test",
                name="Deepseek Api Key",
                description="Deepseek Api Key",
                provider=ModelProvider.DEEPSEEK,
                api_key="deepseek_api_key",
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

        self._model_credential_prefix = "/api/model-credentials"

        self._http_client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout)
        )

    async def close(self):
        """Closes the underlying HTTP connections.

        Examples:
            ```python
            await client.model_credentials.close()
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

    async def create(self, org_id: str, request: ModelCredentialCreateParams) -> ModelCredentialResponseParams:
        """Creates a new model credential in Graxon.

        Args:
            org_id: The unique identifier of the organization to create the
                model credential under.
            request: The parameters for the model credential to create,
                including its `name`, `description`, `provider`, and `api_key`.

        Returns:
            ModelCredentialResponseParams: The newly created model credential,
            including its assigned `id`.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            credential = await client.model_credentials.create(
                org_id="test",
                request=ModelCredentialCreateParams(
                    org_id="test",
                    name="Deepseek Api Key",
                    description="Deepseek Api Key",
                    provider=ModelProvider.DEEPSEEK,
                    api_key="deepseek_api_key",
                ),
            )
            print(credential.id)
            ```
        """
        payload = request.model_dump()
        res_data = await self._request("POST", f"{self._model_credential_prefix}/{org_id}/create", json=payload)

        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")

        return ModelCredentialResponseParams(**data)

    async def list_by_provider(self, org_id: str, provider: ModelProvider) -> list[ModelCredentialResponseParams]:
        """Lists all model credentials for an organization filtered by provider.

        Args:
            org_id: The unique identifier of the organization to list model
                credentials for.
            provider: The provider to filter by (e.g. `ModelProvider.DEEPSEEK`).

        Returns:
            list[ModelCredentialResponseParams]: All model credentials matching
            the given provider.

        Raises:
            GraxonAPIError: If the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            credentials = await client.model_credentials.list_by_provider(
                org_id="test", provider=ModelProvider.DEEPSEEK
            )
            for credential in credentials:
                print(credential.id, credential.name)
            ```
        """
        res_data = await self._request("GET", f"{self._model_credential_prefix}/{org_id}/get/all/provider/{provider.value}")

        # Handle the nested {"data": {"data": [...]}} structure
        wrapper_data = res_data.get("data") or {}
        list_data = wrapper_data.get("data") or []

        return [ModelCredentialResponseParams(**item) for item in list_data]

    async def get(self, org_id: str, model_credential_id: uuid.UUID) -> ModelCredentialResponseParams:
        """Retrieves a specific model credential by ID.

        Args:
            org_id: The unique identifier of the organization the model
                credential belongs to.
            model_credential_id: The unique identifier of the model credential to fetch.

        Returns:
            ModelCredentialResponseParams: The matching model credential.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            credential = await client.model_credentials.get(
                org_id="test", model_credential_id=create_response.id
            )
            print(credential.name)
            ```
        """
        res_data = await self._request("GET", f"{self._model_credential_prefix}/{org_id}/get/{model_credential_id}")
        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return ModelCredentialResponseParams(**data)

    async def delete(self, org_id: str, model_credential_id: uuid.UUID) -> Any:
        """Deletes a model credential by ID.

        Args:
            org_id: The unique identifier of the organization the model
                credential belongs to.
            model_credential_id: The unique identifier of the model credential to delete.

        Returns:
            Any: The raw API response confirming deletion.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            result = await client.model_credentials.delete(
                org_id="test", model_credential_id=create_response.id
            )
            print(result)
            ```
        """
        return await self._request("DELETE", f"{self._model_credential_prefix}/{org_id}/delete/{model_credential_id}")
