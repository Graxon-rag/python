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
        res_data = await self._request("GET", f"{self._document_prefix}/{org_id}/projects/{project_id}/get/all")
        list_data = res_data.get("data", {}).get("data", [])

        return [DocumentGetParams(**item) for item in list_data]

    async def get(self, org_id: str, project_id: uuid.UUID, document_id: uuid.UUID) -> DocumentGetParams:
        res_data = await self._request("GET", f"{self._document_prefix}/{org_id}/projects/{project_id}/get/{document_id}")
        data = res_data.get("data", {})
        if not data:
            raise GraxonAPIError("Graxon API Error: Response missing 'data' payload")
        return DocumentGetParams(**data)

    async def delete(self, org_id: str, project_id: uuid.UUID, document_id: uuid.UUID) -> Dict[str, Any]:
        res_data = await self._request("DELETE", f"{self._document_prefix}/{org_id}/projects/{project_id}/delete/{document_id}")
        return res_data.get("data", {})

    async def process(self, org_id: str, project_id: uuid.UUID, document_id: uuid.UUID) -> bool:
        res_data = await self._request("POST", f"{self._document_prefix}/{org_id}/projects/{project_id}/process/{document_id}")
        return res_data.get("status_code") == 202

    async def get_signed_url(self, org_id: str, project_id: uuid.UUID, bucket: str, key: str) -> DocumentResponseSignedUrlParams:
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
        """
        Handles streaming a large local file directly to S3/MinIO via presigned URLs.
        Includes local state tracking to resume automatically if the upload crashes.
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
