from ..errors import GraxonAPIError, GraxonNetworkError
import httpx


class ModelProvider:
    """Client for listing available Model Providers in the Graxon API.

    Examples:
        ```python
        from graxon.client import GraxonAsyncClient

        client = GraxonAsyncClient(api_key="graxon_api_key", base_url="http://localhost:8888")
        all_providers = await client.model_providers.all_models()
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

        self._model_provider_prefix = "/api/model-providers"

        self._http_client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout)
        )

    async def close(self):
        """Closes the underlying HTTP connections.

        Examples:
            ```python
            await client.model_providers.close()
            ```
        """
        await self._http_client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _request(self, method: str, path: str, **kwargs) -> list[str]:
        """Internal helper to dispatch requests and handle standard API errors."""
        try:
            response = await self._http_client.request(method, path, **kwargs)
            response.raise_for_status()

            res_data = response.json()  # it return [string, string....]
            return res_data

        except httpx.HTTPStatusError as e:
            raise GraxonAPIError(
                f"Graxon API Error {e.response.status_code}: {e.response.text}"
            ) from None

        except httpx.RequestError as e:
            raise GraxonNetworkError(
                f"Failed to communicate with Graxon API: {str(e)}"
            ) from None

    async def all_models(self) -> list[str]:
        """Lists all model providers supported by Graxon, across every model type.

        Returns:
            list[str]: The names of every supported provider.

        Raises:
            GraxonAPIError: If the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            providers = await client.model_providers.all_models()
            print(providers)
            ```
        """
        return await self._request("GET", f"{self._model_provider_prefix}/all")

    async def llm_models(self) -> list[str]:
        """Lists all providers that support LLM models.

        Returns:
            list[str]: The names of providers offering LLM models.

        Raises:
            GraxonAPIError: If the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            providers = await client.model_providers.llm_models()
            print(providers)
            ```
        """
        return await self._request("GET", f"{self._model_provider_prefix}/llm_model")

    async def embedding_models(self) -> list[str]:
        """Lists all providers that support embedding models.

        Returns:
            list[str]: The names of providers offering embedding models.

        Raises:
            GraxonAPIError: If the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            providers = await client.model_providers.embedding_models()
            print(providers)
            ```
        """
        return await self._request("GET", f"{self._model_provider_prefix}/embedding_model")

    async def audio_models(self) -> list[str]:
        """Lists all providers that support audio models.

        Returns:
            list[str]: The names of providers offering audio models.

        Raises:
            GraxonAPIError: If the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            providers = await client.model_providers.audio_models()
            print(providers)
            ```
        """
        return await self._request("GET", f"{self._model_provider_prefix}/audio-model")

    async def video_models(self) -> list[str]:
        """Lists all providers that support video models.

        Returns:
            list[str]: The names of providers offering video models.

        Raises:
            GraxonAPIError: If the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            providers = await client.model_providers.video_models()
            print(providers)
            ```
        """
        return await self._request("GET", f"{self._model_provider_prefix}/video-model")

    async def ocr_models(self) -> list[str]:
        """Lists all providers that support OCR models.

        Returns:
            list[str]: The names of providers offering OCR models.

        Raises:
            GraxonAPIError: If the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            providers = await client.model_providers.ocr_models()
            print(providers)
            ```
        """
        return await self._request("GET", f"{self._model_provider_prefix}/ocr-model")

    async def reranker_models(self) -> list[str]:
        """Lists all providers that support reranker models.

        Returns:
            list[str]: The names of providers offering reranker models.

        Raises:
            GraxonAPIError: If the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            providers = await client.model_providers.reranker_models()
            print(providers)
            ```
        """
        return await self._request("GET", f"{self._model_provider_prefix}/reranker-model")

    async def sparse_models(self) -> list[str]:
        """Lists all providers that support sparse models.

        Returns:
            list[str]: The names of providers offering sparse models.

        Raises:
            GraxonAPIError: If the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            providers = await client.model_providers.sparse_models()
            print(providers)
            ```
        """
        return await self._request("GET", f"{self._model_provider_prefix}/sparse-model")
