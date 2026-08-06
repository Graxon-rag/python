from pydantic import BaseModel, Field
from enum import Enum
import datetime
import uuid


class EmbeddingModelProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    VOYAGE = "voyage"


class EmbeddingModelCreateParams(BaseModel):
    org_id: str = Field(
        description="The organization id of the embedding model",
    )
    name: str = Field(
        description="The name of the embedding model",
    )
    provider: EmbeddingModelProvider = Field(
        description="The provider of the embedding model",
    )
    model_name: str = Field(
        description="The model name of the embedding model",
    )
    model_id: str = Field(
        description="The model id of the embedding model",
    )
    dimension: int = Field(
        description="The dimension of the embedding model",
    )
    description: str = Field(
        description="The description of the embedding model",
    )


class EmbeddingModelResponseParams(BaseModel):
    id: uuid.UUID = Field(
        description="The id of the embedding model",
    )
    org_id: str = Field(
        description="The organization id of the embedding model",
    )
    name: str = Field(
        description="The name of the embedding model",
    )
    provider: EmbeddingModelProvider = Field(
        description="The provider of the embedding model",
    )
    model_name: str = Field(
        description="The model name of the embedding model",
    )
    model_id: str = Field(
        description="The model id of the embedding model",
    )
    dimension: int = Field(
        description="The dimension of the embedding model",
    )
    description: str = Field(
        description="The description of the embedding model",
    )
    created_at: datetime.datetime = Field(
        description="The created at of the embedding model",
    )
    updated_at: datetime.datetime = Field(
        description="The updated at of the embedding model",
    )
