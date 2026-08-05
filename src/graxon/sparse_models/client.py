

class SparseModel:
    def __init__(self, api_key: str | None, base_url: str | None = "http://localhost:8888", timeout: float | None = 120.0):
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
