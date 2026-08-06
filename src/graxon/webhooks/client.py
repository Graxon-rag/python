from .types import WebhookCreateParams, WebhookResponseParams
from ..errors import GraxonAPIError, GraxonNetworkError
from typing import Any, Dict
import httpx
import uuid


class Webhook:
    """Client for managing Webhooks in the Graxon API.

    Examples:
        ```python
        from graxon.client import GraxonAsyncClient
        from graxon.webhooks.types import WebhookCreateParams
        import uuid

        client = GraxonAsyncClient(api_key="graxon_api_key", base_url="http://localhost:8888")
        project_id = uuid.UUID("ba512d16-bbfa-459b-b684-32e8996fd08c")

        webhook = await client.webhooks.create(
            org_id="test",
            project_id=project_id,
            request=WebhookCreateParams(
                org_id="test",
                project_id=project_id,
                name="test",
                url="http://localhost:8080/webhooks/graxon",
                token="xxxxxxxxxxxxxxxxxxx",
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

        self._project_prefix = "/api/webhooks"

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

    async def create(self, org_id: str, project_id: uuid.UUID, request: WebhookCreateParams) -> WebhookResponseParams:
        """Creates a new webhook for a specific project.

        Args:
            org_id: The unique identifier of the organization.
            project_id: The UUID of the project to create the webhook under.
            request: The parameters for the webhook to create, including its
                `name`, `url`, and `token`.

        Returns:
            WebhookResponseParams: The newly created webhook, including its
            assigned `id`.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            create_response = await client.webhooks.create(
                org_id="test",
                project_id=project_id,
                request=WebhookCreateParams(
                    org_id="test",
                    project_id=project_id,
                    name="test",
                    url="http://localhost:8080/webhooks/graxon",
                    token="xxxxxxxxxxxxxxxxxxx",
                ),
            )
            print(create_response.id)
            ```
        """
        payload = request.model_dump(mode='json')
        res_data = await self._request("POST", f"{self._project_prefix}/{org_id}/{project_id}/create", json=payload)
        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return WebhookResponseParams(**data)

    async def get(self, org_id: str, project_id: uuid.UUID, webhook_id: uuid.UUID) -> WebhookResponseParams:
        """Retrieves a specific webhook by its ID.

        Args:
            org_id: The unique identifier of the organization.
            project_id: The UUID of the project the webhook belongs to.
            webhook_id: The UUID of the webhook to retrieve.

        Returns:
            WebhookResponseParams: The requested webhook details.

        Raises:
            GraxonAPIError: If the webhook is not found, or the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            get_response = await client.webhooks.get(
                org_id="test",
                project_id=project_id,
                webhook_id=create_response.id,
            )
            print(get_response.name)
            ```
        """
        res_data = await self._request("GET", f"{self._project_prefix}/{org_id}/{project_id}/get/{webhook_id}")
        data = res_data.get("data")
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return WebhookResponseParams(**data)

    async def list(self, org_id: str, project_id: uuid.UUID) -> list[WebhookResponseParams]:
        """Retrieves a list of all webhooks for a specific project.

        Args:
            org_id: The unique identifier of the organization.
            project_id: The UUID of the project.

        Returns:
            list[WebhookResponseParams]: A list of all webhooks belonging to the project.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            list_response = await client.webhooks.list(
                org_id="test",
                project_id=project_id,
            )
            for webhook in list_response:
                print(webhook.name)
            ```
        """
        res_data = await self._request("GET", f"{self._project_prefix}/{org_id}/{project_id}/list")

        # Handle the nested {"data": {"data": [...]}} structure
        wrapper_data = res_data.get("data") or {}
        list_data = wrapper_data.get("data") or []

        return [WebhookResponseParams(**item) for item in list_data]

    async def delete(self, org_id: str, project_id: uuid.UUID, webhook_id: uuid.UUID) -> Dict[str, Any]:
        """Deletes a specific webhook by its ID.

        Args:
            org_id: The unique identifier of the organization.
            project_id: The UUID of the project the webhook belongs to.
            webhook_id: The UUID of the webhook to delete.

        Returns:
            Dict[str, Any]: A dictionary containing the response payload confirming deletion.

        Raises:
            GraxonAPIError: If the webhook is not found, or the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            delete_response = await client.webhooks.delete(
                org_id="test",
                project_id=project_id,
                webhook_id=create_response.id,
            )
            print(delete_response)
            ```
        """
        return await self._request("DELETE", f"{self._project_prefix}/{org_id}/{project_id}/delete/{webhook_id}")
