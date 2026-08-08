from .types import DocumentMultipartUploadPartParams, PresignedUrlRequestParams, CompleteMultipartUploadParams
from ..errors import GraxonAPIError, GraxonNetworkError
from typing import Any, Dict, List
import tempfile
import logging
import httpx
import uuid
import os
import re

logger = logging.getLogger(__name__)


class Document:
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

    async def list(self, org_id: str, project_id: uuid.UUID) -> List[Dict[str, Any]]:
        res_data = await self._request("GET", f"{self._document_prefix}/{org_id}/projects/{project_id}/get/all")
        return res_data.get("data", {}).get("data", [])

    async def get(self, org_id: str, project_id: uuid.UUID, document_id: uuid.UUID) -> Dict[str, Any]:
        res_data = await self._request("GET", f"{self._document_prefix}/{org_id}/projects/{project_id}/get/{document_id}")
        return res_data.get("data", {})

    async def delete(self, org_id: str, project_id: uuid.UUID, document_id: uuid.UUID) -> bool:
        res_data = await self._request("DELETE", f"{self._document_prefix}/{org_id}/projects/{project_id}/delete/{document_id}")
        return res_data.get("data", {}).get("success", False)

    async def process(self, org_id: str, project_id: uuid.UUID, document_id: uuid.UUID) -> bool:
        res_data = await self._request("POST", f"{self._document_prefix}/{org_id}/projects/{project_id}/process/{document_id}")
        return res_data.get("status_code") == 202

    async def get_signed_url(self, org_id: str, project_id: uuid.UUID, bucket: str, key: str) -> Dict[str, Any]:
        res_data = await self._request(
            "GET", 
            f"{self._document_prefix}/{org_id}/projects/{project_id}/get-signed-url",
            params={"bucket": bucket, "key": key}
        )
        return res_data.get("data", {})

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

    async def upload(self, org_id: str, project_id: uuid.UUID, file_path: str, document_id: uuid.UUID = uuid.uuid4(), is_ocr_needed: bool = False, chunk_size_in_mb: int = 10) -> Dict[str, Any]:
        """
        Handles splitting a large local file into temporary chunks, acquiring presigned URLs, 
        uploading them, and completing the upload. Automatically cleans up tmp files on success or error.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_name = self._sanitize_file_name(os.path.basename(file_path))
        file_size = os.path.getsize(file_path)
        chunk_size = chunk_size_in_mb * 1024 * 1024  # chunks (MinIO/S3 minimum)

        logger.info(f"Starting multipart upload for {file_name} ({file_size} bytes)")

        # Using TemporaryDirectory guarantees that the tmp folder and all chunk files 
        # inside it are deleted when the block exits (even if an exception is raised).
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                # Read the main file and split it into chunks stored in the tmp directory
                chunk_files = []
                logger.info(f"Splitting {file_name} into {chunk_size} byte chunks in {tmp_dir}")

                with open(file_path, 'rb') as main_file:
                    part_num = 1
                    while True:
                        chunk_data = main_file.read(chunk_size)
                        if not chunk_data:
                            break

                        chunk_path = os.path.join(tmp_dir, f"part_{part_num}.tmp")
                        with open(chunk_path, 'wb') as chunk_file:
                            chunk_file.write(chunk_data)

                        chunk_files.append((part_num, chunk_path))
                        part_num += 1

                # Initialize the multipart upload with the backend
                init_data = await self._init_multipart(org_id, project_id, document_id, file_name)
                upload_id = init_data.get("upload_id")
                key = init_data.get("key")

                if not upload_id or not key:
                    raise GraxonAPIError("Initialization failed: Missing upload_id or key in response")

                completed_parts = []

                # Iterate over the saved tmp files and upload them
                # Using a separate raw httpx client since S3/MinIO doesn't use the Graxon API envelope
                async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as s3_client:
                    for part_num, chunk_path in chunk_files:

                        # Get the presigned URL for this specific part
                        url_data = await self._get_multipart_presigned_url(org_id, project_id, document_id, upload_id, key, part_num)
                        presigned_url = url_data.get("url")
                        if not presigned_url:
                            raise GraxonAPIError(f"Failed to get presigned URL for part {part_num}")

                        logger.info(f"Uploading part {part_num}/{len(chunk_files)} for {file_name}")

                        # Read the chunk from the tmp file into memory first to avoid httpx sync/async conflict
                        with open(chunk_path, 'rb') as chunk_to_upload:
                            chunk_bytes = chunk_to_upload.read()

                        # Upload the raw bytes to the presigned URL
                        s3_response = await s3_client.put(presigned_url, content=chunk_bytes)
                        s3_response.raise_for_status()

                        # Extract the ETag from the S3 response headers (required for completion)
                        etag = s3_response.headers.get("etag") or s3_response.headers.get("ETag")
                        if not etag:
                            raise GraxonAPIError(f"Failed to get ETag for part {part_num}")

                        # S3 sometimes returns ETags with quotes, they must be included
                        completed_parts.append(
                            DocumentMultipartUploadPartParams(etag=etag, part_number=part_num)
                        )

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

                return await self._complete_multipart(org_id, project_id, document_id, complete_payload)

            except Exception as e:
                logger.error(f"Multipart upload failed for {file_name}. Error: {str(e)}. Cleaning up temporary files.")
                raise e
