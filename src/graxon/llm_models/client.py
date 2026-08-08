from .types import LLMModelCreateParams, LLMModelResponseParams, LLMModelProvider
from ..errors import GraxonAPIError, GraxonNetworkError
from typing import Any, Dict
import httpx
import uuid


class LLMModel:
    """Client for managing LLM Models in the Graxon API.

    Examples:
        ```python
        from graxon.client import GraxonAsyncClient

        client = GraxonAsyncClient(api_key="graxon_api_key", base_url="http://localhost:8888")
        model = await client.llm_models.create(
            org_id="test",
            request=LLMModelCreateParams(
                org_id="test",
                name="OpenAI Test Model",
                model_name="text-davinci-003",
                model_id="text-davinci-003",
                provider=LLMModelProvider.OPENAI,
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

        self._llm_model_prefix = "/api/llm-models"

        self._http_client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout)
        )

    async def close(self):
        """Closes the underlying HTTP connections.

        Examples:
            ```python
            await client.llm_models.close()
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

    async def create(self, org_id: str, request: LLMModelCreateParams) -> LLMModelResponseParams:
        """Creates a new LLM model for an organization.

        Args:
            org_id: The unique identifier of the organization to create the
                LLM model under.
            request: The parameters for the LLM model to create, including its
                `name`, `model_name`, `model_id`, `provider`, and `description`.

        Returns:
            LLMModelResponseParams: The newly created LLM model, including its
            assigned `id`.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            model = await client.llm_models.create(
                org_id="test",
                request=LLMModelCreateParams(
                    org_id="test",
                    name="OpenAI Test Model",
                    model_name="text-davinci-003",
                    model_id="text-davinci-003",
                    provider=LLMModelProvider.OPENAI,
                    description="OpenAI Test Model",
                ),
            )
            print(model.id)
            ```
        """
        payload = request.model_dump(mode='json')
        res_data = await self._request("POST", f"{self._llm_model_prefix}/{org_id}/create", json=payload)
        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return LLMModelResponseParams(**data)

    async def create_multiple(self, org_id: str, request: list[LLMModelCreateParams]) -> dict[str, Any]:
        """Creates multiple LLM models for an organization in a single request.

        Args:
            org_id: The unique identifier of the organization to create the
                LLM models under.
            request: A list of parameters, one per LLM model to create.

        Returns:
            dict[str, Any]: The raw API response payload describing the created
            LLM models.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            result = await client.llm_models.create_multiple(
                org_id="test",
                request=[
                    LLMModelCreateParams(
                        org_id="test",
                        name="DEEPSEEKTest Model",
                        model_name="DEEPSEEK Chat",
                        model_id="deepseek-chat",
                        provider=LLMModelProvider.DEEPSEEK,
                        description="DEEPSEEK Test Model",
                    ),
                    LLMModelCreateParams(
                        org_id="test",
                        name="GEMINI Test Model",
                        model_name="gemini 2.5 pro",
                        model_id="gemini-2.5-pro",
                        provider=LLMModelProvider.GEMINI,
                        description="GEMINI Test Model",
                    ),
                ],
            )
            print(result)
            ```
        """
        payload = [item.model_dump(mode='json') for item in request]
        res_data = await self._request("POST", f"{self._llm_model_prefix}/{org_id}/create-multiple", json=payload)

        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")

        return data

    async def get(self, org_id: str, llm_model_id: uuid.UUID) -> LLMModelResponseParams:
        """Retrieves a specific LLM model by ID.

        Args:
            org_id: The unique identifier of the organization the LLM model
                belongs to.
            llm_model_id: The unique identifier of the LLM model to fetch.

        Returns:
            LLMModelResponseParams: The matching LLM model.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            model = await client.llm_models.get(org_id="test", llm_model_id=create_response.id)
            print(model.name)
            ```
        """
        res_data = await self._request("GET", f"{self._llm_model_prefix}/{org_id}/get/{llm_model_id}")
        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return LLMModelResponseParams(**data)

    async def list_by_provider(self, org_id: str, provider: LLMModelProvider) -> list[LLMModelResponseParams]:
        """Lists all LLM models for an organization filtered by provider.

        Args:
            org_id: The unique identifier of the organization to list LLM
                models for.
            provider: The provider to filter by (e.g. `LLMModelProvider.OPENAI`).

        Returns:
            list[LLMModelResponseParams]: All LLM models matching the given
            provider.

        Raises:
            GraxonAPIError: If the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            models = await client.llm_models.list_by_provider(
                org_id="test", provider=LLMModelProvider.OPENAI
            )
            for model in models:
                print(model.id, model.name)
            ```
        """
        res_data = await self._request("GET", f"{self._llm_model_prefix}/{org_id}/get/all/provider/{provider.value}")
        # Handle the nested {"data": {"data": [...]}} structure
        wrapper_data = res_data.get("data") or {}
        list_data = wrapper_data.get("data") or []

        return [LLMModelResponseParams(**item) for item in list_data]

    async def delete(self, org_id: str, llm_model_id: uuid.UUID) -> dict[str, Any]:
        """Deletes an LLM model by ID.

        Args:
            org_id: The unique identifier of the organization the LLM model
                belongs to.
            llm_model_id: The unique identifier of the LLM model to delete.

        Returns:
            dict[str, Any]: The raw API response confirming deletion.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            result = await client.llm_models.delete(org_id="test", llm_model_id=create_response.id)
            print(result)
            ```
        """
        res_data = await self._request("DELETE", f"{self._llm_model_prefix}/{org_id}/delete/{llm_model_id}")
        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return data
