from .types import ProjectConfigUpdateParams, ProjectConfigGetParams
from ..errors import GraxonAPIError, GraxonNetworkError
from typing import Any, Dict
import httpx
import uuid


class ProjectConfig:
    """Client for managing Project Configurations in the Graxon API.

    Project Configurations dictate which AI models (LLMs, Embeddings, OCR, Rerankers) 
    and feature flags (Graph DB, Sparse Retrieval) are active for a specific project.

    Examples:
        ```python
        from graxon.client import GraxonAsyncClient
        from graxon.project_configs.types import ProjectConfigUpdateParams
        import uuid

        client = GraxonAsyncClient(api_key="graxon_api_key", base_url="http://localhost:8888")
        project_id = uuid.UUID("e538c04e-22c0-41a4-a1f6-49b91183e0bb")
        config_id = uuid.UUID("6f3121ef-68d0-4ab5-80d5-973ceb3a6b1a")

        # Fetch current config
        current_config = await client.project_configs.get(
            org_id="test",
            project_id=project_id,
            config_id=config_id,
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
        """Retrieves a specific project configuration by its config ID.

        Args:
            org_id: The unique identifier of the organization.
            project_id: The UUID of the project.
            config_id: The UUID of the specific project configuration to retrieve.

        Returns:
            ProjectConfigGetParams: The requested project configuration details.

        Raises:
            GraxonAPIError: If the configuration is not found, or the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            get_response = await client.project_configs.get(
                org_id="test",
                project_id=project_id,
                config_id=config_id,
            )
            print(get_response.llm_model_id)
            ```
        """
        res_data = await self._request("GET", f"{self._project_prefix}/{org_id}/{project_id}/get/{config_id}")
        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return ProjectConfigGetParams(**data)

    async def get_by_project(self, org_id: str, project_id: uuid.UUID) -> ProjectConfigGetParams:
        """Retrieves the project configuration associated with a specific project ID.

        Args:
            org_id: The unique identifier of the organization.
            project_id: The UUID of the project whose configuration you want to retrieve.

        Returns:
            ProjectConfigGetParams: The requested project configuration details.

        Raises:
            GraxonAPIError: If the configuration is not found, or the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            get_response = await client.project_configs.get_by_project(
                org_id="test",
                project_id=project_id,
            )
            print(get_response.id)
            ```
        """
        res_data = await self._request("GET", f"{self._project_prefix}/{org_id}/{project_id}")
        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return ProjectConfigGetParams(**data)

    async def details_by_project(self, org_id: str, project_id: uuid.UUID) -> Any:
        """Retrieves detailed project configuration and status information by project ID.

        Args:
            org_id: The unique identifier of the organization.
            project_id: The UUID of the project.

        Returns:
            Any: The detailed project configuration data, parsed into `ProjectConfigGetParams`.

        Raises:
            GraxonAPIError: If the configuration is not found, or the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            details = await client.project_configs.details_by_project(
                org_id="test",
                project_id=project_id,
            )
            print(details)
            ```
        """
        res_data = await self._request("GET", f"{self._project_prefix}/{org_id}/{project_id}/details")
        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return ProjectConfigGetParams(**data)

    async def update(self, org_id: str, project_id: uuid.UUID, config_id: uuid.UUID, update: ProjectConfigUpdateParams) -> ProjectConfigGetParams:
        """Updates specific fields of a project configuration.

        Args:
            org_id: The unique identifier of the organization.
            project_id: The UUID of the project.
            config_id: The UUID of the project configuration to update.
            update: The parameters to update. Only provided fields will be updated; 
                unset or None fields are excluded from the payload.

        Returns:
            ProjectConfigGetParams: The updated project configuration details.

        Raises:
            GraxonAPIError: If the update fails, or the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            updated_llm_id = uuid.UUID("1658848a-47c2-4900-a3d1-dea27223587d")

            update_response = await client.project_configs.update(
                org_id="test",
                project_id=project_id,
                config_id=config_id,
                update=ProjectConfigUpdateParams(
                   llm_model_id=updated_llm_id,
                   # Any field omitted here remains unchanged
                ),
            )
            print(update_response.llm_model_id)
            ```
        """
        payload = update.model_dump(mode='json', exclude_unset=True, exclude_none=True)
        res_data = await self._request("PUT", f"{self._project_prefix}/{org_id}/{project_id}/update/{config_id}", json=payload)
        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return ProjectConfigGetParams(**data)

    async def delete(self, org_id: str, project_id: uuid.UUID, config_id: uuid.UUID) -> Dict[str, Any]:
        """Deletes a specific project configuration by its config ID.

        Args:
            org_id: The unique identifier of the organization.
            project_id: The UUID of the project.
            config_id: The UUID of the project configuration to delete.

        Returns:
            Dict[str, Any]: A dictionary containing the response payload confirming deletion.

        Raises:
            GraxonAPIError: If the configuration is not found, or the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            delete_response = await client.project_configs.delete(
                org_id="test",
                project_id=project_id,
                config_id=config_id,
            )
            print(delete_response)
            ```
        """
        return await self._request("DELETE", f"{self._project_prefix}/{org_id}/{project_id}/delete/{config_id}")
