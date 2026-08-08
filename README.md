# Graxon Python SDK

> **Graxon**: First Open-Source Agentic Hybrid RAG to eliminate hallucinations through a persistent Knowledge Graph layer.

Graxon combines dense vector search, sparse retrieval, and a structured Knowledge Graph to deliver accurate, traceable, and context-aware answers — at scale, across multiple organizations, projects, and documents.

---

## Table of Contents

- [API & Docs](#api--docs)
- [Install](#install)
- [Examples](#examples)
- [Usage](#usage)
  - [1. Initialization](#1-initialization)
  - [2. Organizations](#2-organizations)
  - [3. Model Credentials](#3-model-credentials)
  - [4. Exploring Available Providers](#4-exploring-available-providers)
  - [5. Managing Models](#5-managing-models)
  - [6. Managing Projects & Configurations](#6-managing-projects--configurations)
    - [6.1. Creating and Managing Projects](#61-creating-and-managing-projects)
    - [6.2. Updating Project Configurations](#62-updating-project-configurations)
  - [7. Webhooks](#7-webhooks)
  - [8. Managing Documents](#8-managing-documents)
- [Error Handling](#error-handling)

---

## API & Docs

- **API References** -> https://www.graxonrag.com/docs/api-reference

---

## Install

```python
pip install graxon
```

## Examples

For more comprehensive scripts, end-to-end use cases, and advanced configurations, please check out our official examples repository:

👉 [Graxon Python Examples (GitHub)](https://github.com/Graxon-rag/python-examples)

---

## Usage

The Graxon Python SDK is fully asynchronous, built on top of `httpx`. You will need to use an `async` environment (like `asyncio`) to interact with the client.

### 1. Initialization

Import the `GraxonAsyncClient` and initialize it with your API key or None for local server.

```python
import asyncio
from graxon.client import GraxonAsyncClient

async def main():
    client = GraxonAsyncClient(
        api_key="your_graxon_api_key",
        base_url="http://localhost:8888", # Optional: Defaults to production URL
        timeout=60.0
    )

    # Your integration logic here...

    # Close the client when done
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

---

### 2. Organizations

Manage the top-level organizations in your Graxon instance.

```python
from graxon.orgs.types import OrganizationCreateParams

# Create Organization
org = await client.orgs.create(
    request=OrganizationCreateParams(
        name="test-org",
        description="Test Organization"
    )
)
print(f"Created Org ID: {org.id}")

# List all Organizations
orgs_list = await client.orgs.list()

# Get or Delete an Organization
fetched_org = await client.orgs.get(org_id=org.id)
await client.orgs.delete(org_id=org.id)
```

---

### 3. Model Credentials

Securely store API keys for different providers (OpenAI, DeepSeek, etc.) at the organization level.

```python
from graxon.model_credentials.types import ModelCredentialCreateParams
from graxon.types import ModelProvider

org_id = "your-org-id"

credential = await client.model_credentials.create(
    org_id=org_id,
    request=ModelCredentialCreateParams(
        org_id=org_id,
        name="Deepseek API Key",
        description="Production Deepseek Key",
        provider=ModelProvider.DEEPSEEK,
        api_key="your_actual_deepseek_api_key"
    )
)

# List credentials by provider
creds = await client.model_credentials.list_by_provider(
    org_id=org_id,
    provider=ModelProvider.DEEPSEEK
)
```

---

### 4. Exploring Available Providers

You can dynamically fetch the supported models and providers available on your Graxon instance.

```python
# List all globally supported models
all_models = await client.model_providers.all_models()

# Filter by specific model types
llm_providers = await client.model_providers.llm_models()
embedding_providers = await client.model_providers.embedding_models()
sparse_providers = await client.model_providers.sparse_models()
reranker_providers = await client.model_providers.reranker_models()
audio_providers = await client.model_providers.audio_models()
video_providers = await client.model_providers.video_models()
ocr_providers = await client.model_providers.ocr_models()
```

---

### 5. Managing Models

Graxon allows you to configure specific models for different AI tasks. Standard CRUD operations (create, create_multiple, get, list_by_provider, delete) are available across all model clients.

#### LLM Models

```python
from graxon.llm_models.types import LLMModelCreateParams, LLMModelProvider

llm_model = await client.llm_models.create(
    org_id=org_id,
    request=LLMModelCreateParams(
        org_id=org_id,
        name="OpenAI GPT-4",
        model_name="gpt-4",
        model_id="gpt-4",
        provider=LLMModelProvider.OPENAI,
        description="Primary reasoning model"
    )
)
```

#### Embedding Models

```python
from graxon.embedding_models.types import EmbeddingModelCreateParams, EmbeddingModelProvider

embedding_model = await client.embedding_models.create(
    org_id=org_id,
    request=EmbeddingModelCreateParams(
        org_id=org_id,
        name="OpenAI Text Embedding 3",
        model_name="text-embedding-3-small",
        model_id="text-embedding-3-small",
        provider=EmbeddingModelProvider.OPENAI,
        dimension=1536,
        description="Dense vector model"
    )
)
```

#### Audio Models

```python
from graxon.audio_models.types import AudioModelCreateParams, AudioModelProvider

audio_model = await client.audio_models.create(
    org_id=org_id,
    request=AudioModelCreateParams(
        org_id=org_id,
        name="Deepgram Transcription",
        model_name="en-US_BroadbandModel",
        model_id="en-US_BroadbandModel",
        provider=AudioModelProvider.DEEPGRAM,
        description="High-speed audio transcription"
    )
)
```

#### Video Models

```python
from graxon.video_models.types import VideoModelCreateParams, VideoModelProvider

video_model = await client.video_models.create(
    org_id=org_id,
    request=VideoModelCreateParams(
        org_id=org_id,
        name="TwelveLabs Video",
        provider=VideoModelProvider.TWELVELABS,
        model_name="TwelveLabs Marengo",
        model_id="twelvelabs-marengo-2.6",
    ),
)
```

#### OCR Models

```python
from graxon.ocr_models.types import OCRModelCreateParams, OCRModelProvider

ocr_model = await client.ocr_models.create(
    org_id=org_id,
    request=OCRModelCreateParams(
        org_id=org_id,
        name="Mistral OCR Model",
        provider=OCRModelProvider.MISTRAL,
        model_name="Mistral OCR",
        model_id="mistral-ocr-latest",
    ),
)
```

#### Reranker Models

```python
from graxon.reranker_models.types import RerankerModelCreateParams, RerankerModelProvider, RerankerModelProviderType

reranker = await client.reranker_models.create(
    org_id=org_id,
    request=RerankerModelCreateParams(
        org_id=org_id,
        name="Cohere Reranker",
        provider_type=RerankerModelProviderType.CLOUD,
        provider=RerankerModelProvider.COHERE,
        model_name="Cohere Rerank",
        model_id="rerank-english-v3.0",
        size_in_gb=0.0,
    ),
)
```

#### Sparse Text Models

```python
from graxon.sparse_models.types import SparseModelCreateParams, SparseModelProvider, SparseModelProviderType

sparse_model = await client.sparse_models.create(
    org_id=org_id,
    request=SparseModelCreateParams(
        org_id=org_id,
        name="Pinecone Sparse",
        provider_type=SparseModelProviderType.CLOUD,
        provider=SparseModelProvider.PINECONE,
        model_name="Pinecone Sparse English",
        model_id="pinecone-sparse-english-v0",
        size_in_gb=0.0,
    ),
)
```

---

### 6. Managing Projects & Configurations

In Graxon, a **Project** acts as a logical container for your documents, knowledge graph, and vector stores. Every project is tied to a **Project Configuration**, which dictates exactly which AI models (LLMs, Embeddings, OCR, Rerankers) and features (e.g., Graph DB, Sparse Retrieval) are active for that specific project.

#### 6.1. Creating and Managing Projects

When you create a new project, you must supply a `ProjectConfigCreateParams` object mapping your desired models and credentials to the project.

```python
from graxon.projects.types import ProjectCreateParams
from graxon.project_configs.types import ProjectConfigCreateParams
import uuid

org_id = "your-org-id"

# 1. Define the Project Configuration
# Note: You get these UUIDs after creating the respective models and credentials.
project_config = ProjectConfigCreateParams(
    # Feature Flags
    graph_db_enable=True,
    sparse_embedding_enable=True,
    reranker_enable=True,
    llm_tag_extraction_enable=True,

    # Model Mappings
    llm_model_id=uuid.UUID("713c1e08..."),
    llm_model_credential_id=uuid.UUID("e432613d..."),

    embedding_model_id=uuid.UUID("cea2db34..."),
    embedding_model_credential_id=uuid.UUID("8a6212d6..."),

    sparse_text_model_id=uuid.UUID("91077ca9..."),
    sparse_text_model_credential_id=uuid.UUID("aea0a0bb..."),

    reranker_model_id=uuid.UUID("514618b6..."),
    reranker_model_credential_id=uuid.UUID("dc631edb..."),

    ocr_model_id=uuid.UUID("862fcfc3..."),
    ocr_model_credential_id=uuid.UUID("abff39aa..."),

    audio_model_id=uuid.UUID("27007f53..."),
    audio_model_credential_id=uuid.UUID("7849d7eb..."),

    video_model_id=uuid.UUID("6e23df10..."),
    video_model_credential_id=uuid.UUID("5709ed71..."),
)

# 2. Create the Project
new_project = await client.projects.create(
    org_id=org_id,
    request=ProjectCreateParams(
        org_id=org_id,
        name="Knowledge Base v1",
        description="Production RAG pipeline",
        config=project_config,
        project_metadata={"environment": "production", "department": "HR"}
    )
)
print(f"Project Created! ID: {new_project.id}")

# 3. Retrieve a Project
project = await client.projects.get(org_id=org_id, project_id=new_project.id)

# 4. List all Projects in the Organization
all_projects = await client.projects.list(org_id=org_id)

# 5. Delete a Project
await client.projects.delete(org_id=org_id, project_id=new_project.id)
```

#### 6.2. Updating Project Configurations

As your needs evolve, you might want to swap out models (e.g., upgrading from GPT-3.5 to GPT-4) or toggle features without deleting the project. You can manage this via the `project_configs` client.

```python
from graxon.project_configs.types import ProjectConfigUpdateParams
import uuid

project_id = uuid.UUID("e538c04e-22c0-41a4-a1f6-49b91183e0bb")
config_id = uuid.UUID("6f3121ef-68d0-4ab5-80d5-973ceb3a6b1a") # Fetched from your project details

# 1. Get current configuration
current_config = await client.project_configs.get(
    org_id=org_id,
    project_id=project_id,
    config_id=config_id,
)

# 2. Update specific fields (e.g., swapping the LLM model)
update_response = await client.project_configs.update(
    org_id=org_id,
    project_id=project_id,
    config_id=config_id,
    update=ProjectConfigUpdateParams(
       llm_model_id=uuid.UUID("new-llm-model-uuid"),
       llm_model_credential_id=uuid.UUID("new-credential-uuid")
       # Any field omitted here will remain unchanged
    ),
)
print("Configuration updated successfully.")
```

---

### 7. Webhooks

Webhooks are scoped to a specific `project_id` within your organization. Use them to receive real-time asynchronous updates.

```python
from graxon.webhooks.types import WebhookCreateParams
import uuid

project_id = uuid.UUID("ba512d16-bbfa-459b-b684-32e8996fd08c")

# Create a Webhook
webhook = await client.webhooks.create(
    org_id=org_id,
    project_id=project_id,
    request=WebhookCreateParams(
        org_id=org_id,
        project_id=project_id,
        name="Document Pipeline Webhook",
        url="https://your-domain.com/webhooks/graxon",
        token="your_secure_webhook_token_here",
    ),
)

# List project Webhooks
webhooks = await client.webhooks.list(org_id=org_id, project_id=project_id)
```

---

### 8. Managing Documents

The `documents` client allows you to seamlessly upload large files (PDFs, videos, audio), manage their lifecycle within a project, generate secure access URLs, and trigger data processing pipelines.

**Note on Uploads:** The `upload` method features a built-in, resumable multipart upload system. If a large file upload gets interrupted (e.g., network crash), simply run the exact same `upload` command again, and it will automatically resume from the last successful chunk.

```python

import uuid

project_id = uuid.UUID("e538c04e-22c0-41a4-a1f6-49b91183e0bb")
file_path = "./data/large_dataset.pdf"

# 1. Upload a Document (Resumable Multipart)
upload_response = await client.documents.upload(
    org_id=org_id,
    project_id=project_id,
    file_path=file_path,
    chunk_size_in_mb=15, # Optional: Customize chunk size
    is_ocr_needed=True   # Optional: Flag for OCR requirement
)
print(f"Uploaded Document ID: {upload_response.document_id}")

# 2. Trigger Processing Pipeline (Chunking, Embedding, Knowledge Graph)
await client.documents.process(
    org_id=org_id,
    project_id=project_id,
    document_id=upload_response.document_id
)

# 3. Retrieve Document Details
doc = await client.documents.get(
    org_id=org_id,
    project_id=project_id,
    document_id=upload_response.document_id
)
print(f"File: {doc.name}, Status: {doc.status}")

# 4. Generate a Secure Presigned URL for Download/Viewing
url_response = await client.documents.get_signed_url(
    org_id=org_id,
    project_id=project_id,
    bucket=doc.bucket,
    key=doc.key
)
print(f"Temporary Secure URL: {url_response.signed_url}")

# 5. List all Documents in a Project
documents = await client.documents.list(org_id=org_id, project_id=project_id)

# 6. Delete a Document
await client.documents.delete(
    org_id=org_id,
    project_id=project_id,
    document_id=upload_response.document_id
)

```

---

## Error Handling

The SDK provides custom exceptions allowing you to gracefully catch API or network issues.

```python
from graxon.errors import GraxonAPIError, GraxonNetworkError

try:
    await client.orgs.get(org_id="invalid-id")
except GraxonAPIError as e:
    print(f"API Error (e.g., Not Found, Validation Failed): {e}")
except GraxonNetworkError as e:
    print(f"Network Error (e.g., Connection Refused, Timeout): {e}")
```
