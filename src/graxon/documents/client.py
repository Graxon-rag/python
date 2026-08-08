from .types import DocumentMultipartUploadPartParams, PresignedUrlRequestParams, CompleteMultipartUploadParams, DocumentUploadResponseParams, DocumentResponseSignedUrlParams, DocumentGetParams
from ..errors import GraxonAPIError, GraxonNetworkError
from typing import Any, Dict, List
import tempfile
import logging
import httpx
import uuid
import math
import json
import os
import re

logger = logging.getLogger(__name__)


class Document:
    """Client for managing Documents and file uploads in the Graxon API.

    This client supports robust, resumable multipart uploads for large files 
    and handles document lifecycle management including retrieval, processing, 
    and secure access generation.

    Examples:
        ```python
        from graxon.client import GraxonAsyncClient
        import uuid

        client = GraxonAsyncClient(api_key="graxon_api_key", base_url="http://localhost:8888")
        project_id = uuid.UUID("e538c04e-22c0-41a4-a1f6-49b91183e0bb")

        # Upload a file
        upload_response = await client.documents.upload(
            org_id="test",
            project_id=project_id,
            file_path="./data/large.pdf",
            chunk_size_in_mb=15,
        )
        print(upload_response.document_id)
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

        self._document_prefix = "/api/documents"

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
        """Internal helper to dispatch requests, log them, and handle standard API errors."""
        url = f"{self._base_url.rstrip('/')}{path}"
        logger.info(f"[Graxon API] {method} request to {url}")

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
            logger.error(f"[Graxon API] HTTP Error {e.response.status_code} on {method} {url}: {e.response.text}")
            raise GraxonAPIError(
                f"Graxon API Error {e.response.status_code}: {e.response.text}"
            ) from None

        except httpx.RequestError as e:
            logger.error(f"[Graxon API] Network Error on {method} {url}: {str(e)}")
            raise GraxonNetworkError(
                f"Failed to communicate with Graxon API: {str(e)}"
            ) from None

    async def list(self, org_id: str, project_id: uuid.UUID) -> List[DocumentGetParams]:
        """Retrieves a list of all documents within a specific project.

        Args:
            org_id: The unique identifier of the organization.
            project_id: The UUID of the project.

        Returns:
            List[DocumentGetParams]: A list of all documents belonging to the project.

        Raises:
            GraxonAPIError: If the API responds with an error or unexpected payload.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            list_response = await client.documents.list(
                org_id="test", 
                project_id=project_id
            )
            for doc in list_response:
                print(doc.name, doc.id)
            ```
        """
        res_data = await self._request("GET", f"{self._document_prefix}/{org_id}/projects/{project_id}/get/all")
        list_data = res_data.get("data", {}).get("data", [])

        return [DocumentGetParams(**item) for item in list_data]

    async def get(self, org_id: str, project_id: uuid.UUID, document_id: uuid.UUID) -> DocumentGetParams:
        """Retrieves the details and metadata of a specific document by its ID.

        Args:
            org_id: The unique identifier of the organization.
            project_id: The UUID of the project the document belongs to.
            document_id: The UUID of the document to retrieve.

        Returns:
            DocumentGetParams: The requested document details.

        Raises:
            GraxonAPIError: If the document is not found, or the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            get_response = await client.documents.get(
                org_id="test", 
                project_id=project_id, 
                document_id=upload_response.document_id
            )
            print(get_response.name, get_response.bucket)
            ```
        """
        res_data = await self._request("GET", f"{self._document_prefix}/{org_id}/projects/{project_id}/get/{document_id}")
        data = res_data.get("data", {})
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return DocumentGetParams(**data)

    async def delete(self, org_id: str, project_id: uuid.UUID, document_id: uuid.UUID) -> Dict[str, Any]:
        """Deletes a specific document and its associated data from a project.

        Args:
            org_id: The unique identifier of the organization.
            project_id: The UUID of the project the document belongs to.
            document_id: The UUID of the document to delete.

        Returns:
            Dict[str, Any]: A dictionary containing the response payload confirming deletion.

        Raises:
            GraxonAPIError: If the document is not found, or the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            delete_response = await client.documents.delete(
                org_id="test", 
                project_id=project_id, 
                document_id=upload_response.document_id
            )
            print(delete_response)
            ```
        """
        res_data = await self._request("DELETE", f"{self._document_prefix}/{org_id}/projects/{project_id}/delete/{document_id}")
        return res_data.get("data", {})

    async def process(self, org_id: str, project_id: uuid.UUID, document_id: uuid.UUID) -> bool:
        """Triggers the background data processing pipeline for a specific document.

        This initiates steps such as OCR transcription, text extraction, chunking, 
        embedding, and graph construction depending on the project's configuration.

        Args:
            org_id: The unique identifier of the organization.
            project_id: The UUID of the project.
            document_id: The UUID of the document to process.

        Returns:
            bool: `True` if the processing request was successfully accepted (HTTP 202).

        Raises:
            GraxonAPIError: If the API responds with an error.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            is_processing = await client.documents.process(
                org_id="test", 
                project_id=project_id, 
                document_id=upload_response.document_id
            )
            print(f"Processing started: {is_processing}")
            ```
        """
        res_data = await self._request("POST", f"{self._document_prefix}/{org_id}/projects/{project_id}/process/{document_id}")
        return res_data.get("status_code") == 202

    async def get_signed_url(self, org_id: str, project_id: uuid.UUID, bucket: str, key: str) -> DocumentResponseSignedUrlParams:
        """Generates a temporary presigned URL to securely access or download the document file.

        Args:
            org_id: The unique identifier of the organization.
            project_id: The UUID of the project.
            bucket: The storage bucket where the file resides (retrieved via `get()`).
            key: The object key/path of the file in the bucket (retrieved via `get()`).

        Returns:
            DocumentResponseSignedUrlParams: Contains the temporary `signed_url`.

        Raises:
            GraxonAPIError: If the API fails to generate the URL.
            GraxonNetworkError: If the request fails to reach the API.

        Examples:
            ```python
            # First, fetch document details to get the bucket and key
            doc = await client.documents.get(org_id, project_id, document_id)

            signed_url_response = await client.documents.get_signed_url(
                org_id="test", 
                project_id=project_id, 
                bucket=doc.bucket, 
                key=doc.key
            )
            print(signed_url_response.signed_url)
            ```
        """
        res_data = await self._request(
            "GET", 
            f"{self._document_prefix}/{org_id}/projects/{project_id}/get-signed-url",
            params={"bucket": bucket, "key": key}
        )
        signed_url = res_data.get("data", {}).get("signed_url", None)

        if not signed_url:
            raise GraxonAPIError("Graxon API Error: Response missing 'signed_url' payload")

        return DocumentResponseSignedUrlParams(signed_url=signed_url)

    async def _init_multipart(self, org_id: str, project_id: uuid.UUID, document_id: uuid.UUID, file_name: str) -> Dict[str, Any]:
        res_data = await self._request(
            "POST", 
            f"{self._document_prefix}/{org_id}/projects/{project_id}/upload/multipart/{document_id}/init/{file_name}"
        )
        return res_data.get("data", {})

    async def _get_multipart_presigned_url(self, org_id: str, project_id: uuid.UUID, document_id: uuid.UUID, upload_id: str, key: str, part_number: int) -> Dict[str, Any]:
        payload = PresignedUrlRequestParams(upload_id=upload_id, key=key, part_number=part_number)
        res_data = await self._request(
            "POST", 
            f"{self._document_prefix}/{org_id}/projects/{project_id}/upload/multipart/{document_id}/presigned-url",
            json=payload.model_dump(mode="json")
        )
        return res_data.get("data", {})

    async def _complete_multipart(self, org_id: str, project_id: uuid.UUID, document_id: uuid.UUID, payload: CompleteMultipartUploadParams) -> Dict[str, Any]:
        res_data = await self._request(
            "POST", 
            f"{self._document_prefix}/{org_id}/projects/{project_id}/upload/multipart/{document_id}/complete",
            json=payload.model_dump(mode="json")
        )
        return res_data.get("data", {})

    def _sanitize_file_name(self, file_name: str) -> str:
        # Split the filename into base name and extension
        base_name, ext = os.path.splitext(file_name)

        # Strip spaces from ends
        # Replace one or more non-alphanumeric characters with a single '_'
        clean_base = re.sub(r'[^a-zA-Z0-9]+', '_', base_name.strip())

        # Strip trailing or leading underscores
        clean_base = clean_base.strip('_')

        # Fallback if the filename becomes empty (e.g., if it was originally "___ .pdf")
        final_base = clean_base if clean_base else "document"

        return f"{final_base}{ext}"

    async def upload(self, org_id: str, project_id: uuid.UUID, file_path: str, document_id: uuid.UUID = uuid.uuid4(), is_ocr_needed: bool = False, chunk_size_in_mb: int = 10) -> DocumentUploadResponseParams:
        """Uploads a local file directly to storage via a resumable multipart upload.

        This method chunks large files and uploads them efficiently using presigned URLs. 
        It creates a local JSON state file during the upload. If the upload process is 
        interrupted or crashes, re-running the same method will automatically resume 
        from the last successfully uploaded chunk.

        Args:
            org_id: The unique identifier of the organization.
            project_id: The UUID of the project.
            file_path: The absolute or relative local path to the file being uploaded.
            document_id: An optional predefined UUID for the document. Defaults to a newly generated UUID.
            is_ocr_needed: Flag to mark if the document specifically requires OCR processing. Defaults to `False`.
            chunk_size_in_mb: The size of each upload chunk in Megabytes. Defaults to 10.

        Returns:
            DocumentUploadResponseParams: Contains the `document_id` of the successfully uploaded file.

        Raises:
            FileNotFoundError: If the provided `file_path` does not exist locally.
            GraxonAPIError: If upload initialization, presigned URL generation, or final completion fails.
            GraxonNetworkError: If a network issue interrupts the upload of chunks.

        Examples:
            ```python
            upload_response = await client.documents.upload(
                org_id="test",
                project_id=project_id,
                file_path="./data/video.mp4",
                chunk_size_in_mb=15,
            )
            print(f"Upload complete. Document ID: {upload_response.document_id}")
            ```
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_name = self._sanitize_file_name(os.path.basename(file_path))
        file_size = os.path.getsize(file_path)
        chunk_size = chunk_size_in_mb * 1024 * 1024
        total_parts = math.ceil(file_size / chunk_size) if file_size > 0 else 1

        # Determine state file path
        state_file_path = f"{file_path}.graxon_resume.json"
        state = {}

        # Check for existing resume state
        if os.path.exists(state_file_path):
            try:
                with open(state_file_path, 'r') as sf:
                    state = json.load(sf)

                # Invalidate state if the file size changed since the crash
                if state.get("file_size") != file_size:
                    logger.warning("File size changed since last attempt. Starting fresh.")
                    state = {}
                else:
                    logger.info(f"Found resume state. Resuming upload for {file_name}")
            except Exception as e:
                logger.warning(f"Could not read resume state file: {e}. Starting fresh.")
                state = {}

        # Initialize or restore variables
        if not state:
            # New upload initialization
            current_document_id = document_id or uuid.uuid4()
            init_data = await self._init_multipart(org_id, project_id, current_document_id, file_name)

            upload_id = init_data.get("upload_id")
            key = init_data.get("key")

            if not upload_id or not key:
                raise GraxonAPIError("Initialization failed: Missing upload_id or key in response")

            state = {
                "document_id": str(current_document_id),
                "upload_id": upload_id,
                "key": key,
                "file_size": file_size,
                "completed_parts": {}  # Format: {"part_num": "etag"}
            }
        else:
            # Restore from state
            current_document_id = uuid.UUID(state["document_id"])
            upload_id = state["upload_id"]
            key = state["key"]

        completed_parts_dict = state.get("completed_parts", {})

        # Prepare the parts list for the final complete request
        completed_parts = [
            DocumentMultipartUploadPartParams(etag=etag, part_number=int(p_num)) 
            for p_num, etag in completed_parts_dict.items()
        ]

        logger.info(f"Starting/Resuming upload for {file_name} ({file_size} bytes / {total_parts} parts)")

        try:
            # Stream parts directly to S3/MinIO
            async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as s3_client:
                with open(file_path, 'rb') as main_file:

                    for part_num in range(1, total_parts + 1):
                        part_str = str(part_num)

                        # Skip if already uploaded
                        if part_str in completed_parts_dict:
                            logger.info(f"Skipping part {part_num}/{total_parts} (Already uploaded)")
                            continue

                        # Seek to the exact byte offset for this chunk
                        offset = (part_num - 1) * chunk_size
                        main_file.seek(offset)
                        chunk_bytes = main_file.read(chunk_size)

                        if not chunk_bytes:
                            break

                        # Get presigned URL
                        url_data = await self._get_multipart_presigned_url(org_id, project_id, current_document_id, upload_id, key, part_num)
                        presigned_url = url_data.get("url")

                        if not presigned_url:
                            raise GraxonAPIError(f"Failed to get presigned URL for part {part_num}")

                        logger.info(f"Uploading part {part_num}/{total_parts} for {file_name}")

                        # Upload to S3/MinIO
                        s3_response = await s3_client.put(presigned_url, content=chunk_bytes)
                        s3_response.raise_for_status()

                        # Extract ETag
                        etag = s3_response.headers.get("etag") or s3_response.headers.get("ETag")
                        if not etag:
                            raise GraxonAPIError(f"Failed to get ETag for part {part_num}")

                        completed_parts.append(DocumentMultipartUploadPartParams(etag=etag, part_number=part_num))

                        # Save state after successful part upload
                        completed_parts_dict[part_str] = etag
                        state["completed_parts"] = completed_parts_dict
                        with open(state_file_path, 'w') as sf:
                            json.dump(state, sf)

            # Complete the multipart upload
            logger.info(f"Completing multipart upload for {file_name}")
            complete_payload = CompleteMultipartUploadParams(
                upload_id=upload_id,
                key=key,
                file_name=file_name,
                size=file_size,
                is_ocr_needed=is_ocr_needed,
                parts=completed_parts
            )

            result = await self._complete_multipart(org_id, project_id, current_document_id, complete_payload)
            output_document_id = result.get("document_id") or current_document_id

            # Cleanup the resume state file on successful completion
            if os.path.exists(state_file_path):
                os.remove(state_file_path)

            return DocumentUploadResponseParams(document_id=output_document_id)

        except Exception as e:
            logger.error(f"Multipart upload interrupted for {file_name}. Error: {str(e)}.")
            raise e
