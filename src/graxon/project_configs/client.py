from .types import ProjectConfigUpdateParams, ProjectConfigGetParams
from ..errors import GraxonAPIError, GraxonNetworkError
from typing import Any, Dict
import httpx
import uuid


class ProjectConfig:
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

        self._project_prefix = "/api/project-configs"

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

    async def get(self, org_id: str, project_id: uuid.UUID, config_id: uuid.UUID) -> ProjectConfigGetParams:
        """Retrieves a specific project config by config ID."""
        res_data = await self._request("GET", f"{self._project_prefix}/{org_id}/{project_id}/get/{config_id}")
        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return ProjectConfigGetParams(**data)

    async def get_by_project(self, org_id: str, project_id: uuid.UUID) -> ProjectConfigGetParams:
        """Retrieves a specific project config by project ID."""
        res_data = await self._request("GET", f"{self._project_prefix}/{org_id}/{project_id}")
        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return ProjectConfigGetParams(**data)

    async def details_by_project(self, org_id: str, project_id: uuid.UUID) -> Any:
        """Retrieves a specific project config by project ID."""
        res_data = await self._request("GET", f"{self._project_prefix}/{org_id}/{project_id}/details")
        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return ProjectConfigGetParams(**data)

    async def update(self, org_id: str, project_id: uuid.UUID, config_id: uuid.UUID, update: ProjectConfigUpdateParams) -> ProjectConfigGetParams:
        """Updates a specific project config by config ID."""
        payload = update.model_dump(mode='json', exclude_unset=True, exclude_none=True)
        res_data = await self._request("PUT", f"{self._project_prefix}/{org_id}/{project_id}/update/{config_id}", json=payload)
        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return ProjectConfigGetParams(**data)

    async def delete(self, org_id: str, project_id: uuid.UUID, config_id: uuid.UUID) -> Dict[str, Any]:
        """Deletes a project config by config ID."""
        return await self._request("DELETE", f"{self._project_prefix}/{org_id}/{project_id}/delete/{config_id}")
