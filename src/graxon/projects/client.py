from .types import ProjectCreateParams, ProjectResponseParams
from ..errors import GraxonAPIError, GraxonNetworkError
from typing import Any, Dict
import httpx
import uuid


class Project:
    """Client for managing Projects in the Graxon API.

    Projects act as logical containers for documents, graph databases, and vector stores, 
    tied to a specific set of AI models defined via a Project Configuration.

    Examples:
        ```python
        from graxon.client import GraxonAsyncClient
        from graxon.projects.types import ProjectCreateParams
        from graxon.project_configs.types import ProjectConfigCreateParams
        import uuid

        client = GraxonAsyncClient(api_key="graxon_api_key", base_url="http://localhost:8888")

        # Note: project_config requires mapped UUIDs for models and credentials
        project = await client.projects.create(
            org_id="test",
            request=ProjectCreateParams(
                org_id="test",
                name="Test Project",
                description="Test Project Description",
                config=project_config, 
                project_metadata={"environment": "production"}
            )
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
        """Creates a new project for a specific organization.

        Args:
            org_id: The unique identifier of the organization.
            request: The parameters for the project to create, including its
                `name`, `description`, `config` (model mappings), and optional `project_metadata`.

        Returns:
            ProjectResponseParams: The newly created project, including its
            assigned `id`.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            create_response = await client.projects.create(
                org_id="test",
                request=ProjectCreateParams(
                    org_id="test",
                    name="Test Project",
                    description="Test Project Description",
                    config=project_config,
                    project_metadata={"key": "value"}
                )
            )
            print(create_response.id)
            ```
        """
        payload = request.model_dump(mode='json')
        res_data = await self._request("POST", f"{self._project_prefix}/{org_id}/create", json=payload)

        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")

        return ProjectResponseParams(**data)

    async def get(self, org_id: str, project_id: uuid.UUID) -> ProjectResponseParams:
        """Retrieves a specific project by its ID.

        Args:
            org_id: The unique identifier of the organization.
            project_id: The UUID of the project to retrieve.

        Returns:
            ProjectResponseParams: The requested project details.

        Raises:
            GraxonAPIError: If the project is not found, or the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            get_response = await client.projects.get(
                org_id="test",
                project_id=create_response.id,
            )
            print(get_response.name)
            ```
        """
        res_data = await self._request("GET", f"{self._project_prefix}/{org_id}/get/{project_id}")

        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")

        return ProjectResponseParams(**data)

    async def list(self, org_id: str) -> list[ProjectResponseParams]:
        """Retrieves a list of all projects for a specific organization.

        Args:
            org_id: The unique identifier of the organization.

        Returns:
            list[ProjectResponseParams]: A list of all projects belonging to the organization.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            list_response = await client.projects.list(
                org_id="test",
            )
            for project in list_response:
                print(project.name)
            ```
        """
        res_data = await self._request("GET", f"{self._project_prefix}/{org_id}/get/all")

        # Handle the nested {"data": {"data": [...]}} structure
        wrapper_data = res_data.get("data") or {}
        list_data = wrapper_data.get("data") or []

        return [ProjectResponseParams(**item) for item in list_data]

    async def delete(self, org_id: str, project_id: uuid.UUID) -> Dict[str, Any]:
        """Deletes a specific project by its ID.

        Args:
            org_id: The unique identifier of the organization.
            project_id: The UUID of the project to delete.

        Returns:
            Dict[str, Any]: A dictionary containing the response payload confirming deletion.

        Raises:
            GraxonAPIError: If the project is not found, or the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            delete_response = await client.projects.delete(
                org_id="test",
                project_id=create_response.id,
            )
            print(delete_response)
            ```
        """
        return await self._request("DELETE", f"{self._project_prefix}/{org_id}/delete/{project_id}")
