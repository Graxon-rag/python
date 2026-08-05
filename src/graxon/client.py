from .embedding_models import EmbeddingModel
from .reranker_models import RerankerModel
from .sparse_models import SparseModel
from .audio_models import AudioModel
from .video_models import VideoModel
from .ocr_models import OCRModel
from .llm_models import LLMModel
from .orgs import Organization
from .projects import Project
from .webhooks import Webhook


class GraxonAsyncClient:
    """
    Asynchronous client for interacting with the Graxon API.

    This client provides a non-blocking interface to Graxon services using `httpx`.
    It manages connection pooling, authentication, base URL configuration, and 
    request timeouts for all underlying asynchronous API calls.

    Args:
        api_key (str | None): The API key used for authenticating with the Graxon API.
        base_url (str | None, optional): The root URL for the Graxon API. 
            Defaults to "http://localhost:8888".
        timeout (float | None, optional): The maximum time (in seconds) to wait 
            for a request to complete before raising a timeout error. 
            Defaults to 120.0.

    Example:
        >>> import asyncio
        >>> from graxon import GraxonAsyncClient
        >>>
        >>> async def main():
        ...     client = GraxonAsyncClient(
        ...         api_key="grx_12345", 
        ...         base_url="https://api.graxonrag.com/v1"
        ...     )
        ...     response = await client.orgs.get("org_123")
        ...     print(response)
        ...
        >>> asyncio.run(main())
    """
    def __init__(self, api_key: str | None, base_url: str | None = "http://localhost:8888", timeout: float | None = 120.0):
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout

        self.orgs = Organization(api_key, base_url, timeout)
        self.projects = Project(api_key, base_url, timeout)
        self.webhooks = Webhook(api_key, base_url, timeout)

        self.embedding_models = EmbeddingModel(api_key, base_url, timeout)
        self.reranker_models = RerankerModel(api_key, base_url, timeout)
        self.sparse_models = SparseModel(api_key, base_url, timeout)
        self.audio_models = AudioModel(api_key, base_url, timeout)
        self.video_models = VideoModel(api_key, base_url, timeout)
        self.ocr_models = OCRModel(api_key, base_url, timeout)
        self.llm_models = LLMModel(api_key, base_url, timeout)
