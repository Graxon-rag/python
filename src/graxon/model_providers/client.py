from ..errors import GraxonAPIError, GraxonNetworkError
import httpx


class ModelProvider:
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
        """Closes the underlying HTTP connections."""
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
        return await self._request("GET", f"{self._model_provider_prefix}/all")

    async def llm_models(self) -> list[str]:
        return await self._request("GET", f"{self._model_provider_prefix}/llm_model")

    async def embedding_models(self) -> list[str]:
        return await self._request("GET", f"{self._model_provider_prefix}/embedding_model")

    async def audio_models(self) -> list[str]:
        return await self._request("GET", f"{self._model_provider_prefix}/audio-model")

    async def video_models(self) -> list[str]:
        return await self._request("GET", f"{self._model_provider_prefix}/video-model")

    async def ocr_models(self) -> list[str]:
        return await self._request("GET", f"{self._model_provider_prefix}/ocr-model")

    async def reranker_models(self) -> list[str]:
        return await self._request("GET", f"{self._model_provider_prefix}/reranker-model")

    async def sparse_models(self) -> list[str]:
        return await self._request("GET", f"{self._model_provider_prefix}/sparse-model")
