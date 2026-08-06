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
  - [6. Webhooks](#6-webhooks)
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

### 6. Webhooks

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
