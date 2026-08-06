from .types import ProjectCreateParams, ProjectResponseParams
from ..errors import GraxonAPIError, GraxonNetworkError
from typing import Any, Dict
import httpx
import uuid


class Project:
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

        self._project_prefix = "/api/projects"

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

    async def create(self, org_id: str, request: ProjectCreateParams) -> ProjectResponseParams:
        """Creates a new project in Graxon."""
        payload = request.model_dump(mode='json')
        res_data = await self._request("POST", f"{self._project_prefix}/{org_id}/create", json=payload)

        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")

        return ProjectResponseParams(**data)

    async def get(self, org_id: str, project_id: uuid.UUID) -> ProjectResponseParams:
        """Retrieves a specific project by ID."""
        res_data = await self._request("GET", f"{self._project_prefix}/{org_id}/get/{project_id}")

        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")

        return ProjectResponseParams(**data)

    async def list(self, org_id: str) -> list[ProjectResponseParams]:
        """Lists all projects."""
        res_data = await self._request("GET", f"{self._project_prefix}/{org_id}/get/all")

        # Handle the nested {"data": {"data": [...]}} structure
        wrapper_data = res_data.get("data") or {}
        list_data = wrapper_data.get("data") or []

        return [ProjectResponseParams(**item) for item in list_data]

    async def delete(self, org_id: str, project_id: uuid.UUID) -> Dict[str, Any]:
        """Deletes a project by ID."""
        return await self._request("DELETE", f"{self._project_prefix}/{org_id}/delete/{project_id}")
