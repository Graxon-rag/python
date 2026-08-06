from .types import OCRModelCreateParams, OCRModelProvider, OCRModelResponseParams
from ..errors import GraxonAPIError, GraxonNetworkError
from typing import Any, Dict
import httpx
import uuid


class OCRModel:
    """Client for managing OCR Models in the Graxon API.

    Examples:
        ```python
        from graxon.client import GraxonAsyncClient
        from graxon.ocr_models.types import OCRModelCreateParams, OCRModelProvider

        client = GraxonAsyncClient(api_key="graxon_api_key", base_url="http://localhost:8888")
        model = await client.ocr_models.create(
            org_id="test",
            request=OCRModelCreateParams(
                org_id="test",
                name="Datalab Test Model",
                provider=OCRModelProvider.DATALAB,
                model_name="Datalab OCR",
                model_id="datalab-ocr",
                description="Datalab OCR Test Model",
            ),
        )
        ```
    """

    def __init__(
        self,
        api_key: str | None,
        base_url: str = "http://localhost:8888",
        timeout: float | None = 120.0,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout

        headers = {
            "User-Agent": "graxon-python",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self._api_key:
            headers["GRAXON-API-KEY"] = self._api_key

        self._ocr_model_prefix = "/api/ocr-models"

        self._http_client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout),
        )

    async def close(self):
        """Closes the underlying HTTP connections."""
        await self._http_client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Internal helper to dispatch requests and handle standard API errors."""
        try:
            response = await self._http_client.request(
                method,
                path,
                **kwargs,
            )

            response.raise_for_status()

            res_data = response.json()
            success = res_data.get("success")

            if not success:
                error_msg = res_data.get(
                    "message",
                    "API returned success=false",
                )
                raise GraxonAPIError(
                    f"Graxon API Error: {error_msg}"
                )

            return res_data

        except httpx.HTTPStatusError as e:
            raise GraxonAPIError(
                f"Graxon API Error {e.response.status_code}: {e.response.text}"
            ) from None

        except httpx.RequestError as e:
            raise GraxonNetworkError(
                f"Failed to communicate with Graxon API: {str(e)}"
            ) from None

    async def create(
        self,
        org_id: str,
        request: OCRModelCreateParams,
    ) -> OCRModelResponseParams:
        """Creates a new OCR model for an organization.

        Args:
            org_id: The unique identifier of the organization to create the
                OCR model under.
            request: The parameters for the OCR model to create, including its
                `name`, `model_name`, `model_id`, `provider`, and `description`.

        Returns:
            OCRModelResponseParams: The newly created OCR model, including its
            assigned `id`.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            create_response = await client.ocr_models.create(
                org_id="test",
                request=OCRModelCreateParams(
                    org_id="test",
                    name="Datalab Test Model",
                    provider=OCRModelProvider.DATALAB,
                    model_name="Datalab OCR",
                    model_id="datalab-ocr",
                    description="Datalab OCR Test Model",
                ),
            )
            print(create_response.id)
            ```
        """
        payload = request.model_dump()

        res_data = await self._request(
            "POST",
            f"{self._ocr_model_prefix}/{org_id}/create",
            json=payload,
        )

        data = res_data.get("data")

        if not data:
            raise GraxonAPIError(
                "Graxon API Error: Response missing 'data' payload"
            )

        return OCRModelResponseParams(**data)

    async def create_multiple(
        self,
        org_id: str,
        request: list[OCRModelCreateParams],
    ) -> dict[str, Any]:
        """Creates multiple OCR models for an organization in a single request.

        Args:
            org_id: The unique identifier of the organization to create the
                OCR models under.
            request: A list of parameters for the OCR models to create.

        Returns:
            dict[str, Any]: A dictionary containing the response payload with details
            of the created models.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            multiple_create_response = await client.ocr_models.create_multiple(
                org_id="test",
                request=[
                    OCRModelCreateParams(
                        org_id="test",
                        name="Mistral OCR Test Model",
                        provider=OCRModelProvider.MISTRAL,
                        model_name="Mistral OCR",
                        model_id="mistral-ocr-latest",
                        description="Mistral OCR Test Model",
                    ),
                    OCRModelCreateParams(
                        org_id="test",
                        name="LlamaParse Test Model",
                        provider=OCRModelProvider.LLAMAPARSE,
                        model_name="LlamaParse",
                        model_id="llamaparse",
                        description="LlamaParse OCR Test Model",
                    ),
                ],
            )
            print(multiple_create_response)
            ```
        """
        payload = [item.model_dump() for item in request]

        res_data = await self._request(
            "POST",
            f"{self._ocr_model_prefix}/{org_id}/create-multiple",
            json=payload,
        )

        data = res_data.get("data")

        if not data:
            raise GraxonAPIError(
                "Graxon API Error: Response missing 'data' payload"
            )

        return data

    async def get(
        self,
        org_id: str,
        ocr_model_id: uuid.UUID,
    ) -> OCRModelResponseParams:
        """Retrieves a specific OCR model by its ID.

        Args:
            org_id: The unique identifier of the organization.
            ocr_model_id: The UUID of the OCR model to retrieve.

        Returns:
            OCRModelResponseParams: The requested OCR model details.

        Raises:
            GraxonAPIError: If the model is not found, or the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            get_response = await client.ocr_models.get(
                org_id="test",
                ocr_model_id=create_response.id,
            )
            print(get_response.name)
            ```
        """
        res_data = await self._request(
            "GET",
            f"{self._ocr_model_prefix}/{org_id}/get/{ocr_model_id}",
        )

        data = res_data.get("data")

        if not data:
            raise GraxonAPIError(
                "Graxon API Error: Response missing 'data' payload"
            )

        return OCRModelResponseParams(**data)

    async def list_by_provider(
        self,
        org_id: str,
        provider: OCRModelProvider,
    ) -> list[OCRModelResponseParams]:
        """Retrieves a list of OCR models for a specific organization filtered by provider.

        Args:
            org_id: The unique identifier of the organization.
            provider: The OCR provider enum (e.g., DATALAB, MISTRAL, LLAMAPARSE) to filter by.

        Returns:
            list[OCRModelResponseParams]: A list of OCR models belonging to the specified provider.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            list_response = await client.ocr_models.list_by_provider(
                org_id="test",
                provider=OCRModelProvider.DATALAB,
            )
            for model in list_response:
                print(model.name)
            ```
        """
        res_data = await self._request(
            "GET",
            f"{self._ocr_model_prefix}/{org_id}/get/all/provider/{provider.value}",
        )

        # Handle the nested {"data": {"data": [...]}} structure
        wrapper_data = res_data.get("data") or {}
        list_data = wrapper_data.get("data") or []

        return [
            OCRModelResponseParams(**item)
            for item in list_data
        ]

    async def delete(
        self,
        org_id: str,
        ocr_model_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Deletes a specific OCR model by its ID.

        Args:
            org_id: The unique identifier of the organization.
            ocr_model_id: The UUID of the OCR model to delete.

        Returns:
            dict[str, Any]: A dictionary containing the response payload confirming deletion.

        Raises:
            GraxonAPIError: If the model is not found, or the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            delete_response = await client.ocr_models.delete(
                org_id="test",
                ocr_model_id=create_response.id,
            )
            print(delete_response)
            ```
        """
        res_data = await self._request(
            "DELETE",
            f"{self._ocr_model_prefix}/{org_id}/delete/{ocr_model_id}",
        )

        data = res_data.get("data")

        if not data:
            raise GraxonAPIError(
                "Graxon API Error: Response missing 'data' payload"
            )

        return data
