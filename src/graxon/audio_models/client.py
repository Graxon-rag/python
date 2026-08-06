from .types import AudioModelCreateParams, AudioModelResponseParams, AudioModelProvider
from ..errors import GraxonAPIError, GraxonNetworkError
from typing import Any, Dict
import httpx
import uuid


class AudioModel:
    """Client for managing Audio Models in the Graxon API.

    Examples:
        ```python
        from graxon.client import GraxonAsyncClient

        client = GraxonAsyncClient(api_key="graxon_api_key", base_url="http://localhost:8888")
        model = await client.audio_models.create(
            org_id="test",
            request=AudioModelCreateParams(
                org_id="test",
                name="Deepgram Test Model",
                model_name="en-US_BroadbandModel Test Model",
                model_id="en-US_BroadbandModel",
                provider=AudioModelProvider.DEEPGRAM,
                description="Deepgram Test Model",
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

        self._audio_model_prefix = "/api/audio-models"

        self._http_client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout)
        )

    async def close(self):
        """Closes the underlying HTTP connections.

        Examples:
            ```python
            await client.audio_models.close()
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

    async def create(self, org_id: str, request: AudioModelCreateParams) -> AudioModelResponseParams:
        """Creates a new audio model for an organization.

        Args:
            org_id: The unique identifier of the organization to create the
                audio model under.
            request: The parameters for the audio model to create, including
                its `name`, `model_name`, `model_id`, `provider`, and `description`.

        Returns:
            AudioModelResponseParams: The newly created audio model, including
            its assigned `id`.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            model = await client.audio_models.create(
                org_id="test",
                request=AudioModelCreateParams(
                    org_id="test",
                    name="Deepgram Test Model",
                    model_name="en-US_BroadbandModel Test Model",
                    model_id="en-US_BroadbandModel",
                    provider=AudioModelProvider.DEEPGRAM,
                    description="Deepgram Test Model",
                ),
            )
            print(model.id)
            ```
        """
        payload = request.model_dump()
        res_data = await self._request("POST", f"{self._audio_model_prefix}/{org_id}/create", json=payload)

        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")

        return AudioModelResponseParams(**data)

    async def create_multiple(self, org_id: str, request: list[AudioModelCreateParams]) -> dict[str, Any]:
        """Creates multiple audio models for an organization in a single request.

        Args:
            org_id: The unique identifier of the organization to create the
                audio models under.
            request: A list of parameters, one per audio model to create.

        Returns:
            dict[str, Any]: The raw API response payload describing the created
            audio models.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            result = await client.audio_models.create_multiple(
                org_id="test",
                request=[
                    AudioModelCreateParams(
                        org_id="test",
                        name="ELEVENLABS Test Model",
                        model_name="en-ELEVENLABS Test Model",
                        model_id="en-ELEVENLABS",
                        provider=AudioModelProvider.ELEVENLABS,
                        description="ELEVENLABS Test Model",
                    ),
                    AudioModelCreateParams(
                        org_id="test",
                        name="ASSEMBLYAI Test Model",
                        model_name="en-ASSEMBLYAI Test Model",
                        model_id="en-ASSEMBLYAI",
                        provider=AudioModelProvider.ASSEMBLYAI,
                        description="ASSEMBLYAI Test Model",
                    ),
                ],
            )
            print(result)
            ```
        """
        payload = [item.model_dump() for item in request]
        res_data = await self._request("POST", f"{self._audio_model_prefix}/{org_id}/create-multiple", json=payload)

        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")

        return data

    async def get(self, org_id: str, audio_model_id: uuid.UUID) -> AudioModelResponseParams:
        """Retrieves a specific audio model by ID.

        Args:
            org_id: The unique identifier of the organization the audio model
                belongs to.
            audio_model_id: The unique identifier of the audio model to fetch.

        Returns:
            AudioModelResponseParams: The matching audio model.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            model = await client.audio_models.get(org_id="test", audio_model_id=create_response.id)
            print(model.name)
            ```
        """
        res_data = await self._request("GET", f"{self._audio_model_prefix}/{org_id}/get/{audio_model_id}")
        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return AudioModelResponseParams(**data)

    async def list_by_provider(self, org_id: str, provider: AudioModelProvider) -> list[AudioModelResponseParams]:
        """Lists all audio models for an organization filtered by provider.

        Args:
            org_id: The unique identifier of the organization to list audio
                models for.
            provider: The provider to filter by (e.g. `AudioModelProvider.DEEPGRAM`).

        Returns:
            list[AudioModelResponseParams]: All audio models matching the given
            provider.

        Raises:
            GraxonAPIError: If the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            models = await client.audio_models.list_by_provider(
                org_id="test", provider=AudioModelProvider.DEEPGRAM
            )
            for model in models:
                print(model.id, model.name)
            ```
        """
        res_data = await self._request("GET", f"{self._audio_model_prefix}/{org_id}/get/all/provider/{provider.value}")
        # Handle the nested {"data": {"data": [...]}} structure
        wrapper_data = res_data.get("data") or {}
        list_data = wrapper_data.get("data") or []

        return [AudioModelResponseParams(**item) for item in list_data]

    async def delete(self, org_id: str, audio_model_id: uuid.UUID) -> dict[str, Any]:
        """Deletes an audio model by ID.

        Args:
            org_id: The unique identifier of the organization the audio model
                belongs to.
            audio_model_id: The unique identifier of the audio model to delete.

        Returns:
            dict[str, Any]: The raw API response confirming deletion.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            result = await client.audio_models.delete(org_id="test", audio_model_id=create_response.id)
            print(result)
            ```
        """
        res_data = await self._request("DELETE", f"{self._audio_model_prefix}/{org_id}/delete/{audio_model_id}")
        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return data
