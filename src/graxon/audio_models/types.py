from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
import datetime
import uuid


class AudioModelProvider(str, Enum):
    DEEPGRAM = "deepgram"
    GLADIA = "gladia"
    ASSEMBLYAI = "assemblyai"
    GROQ = "groq"
    ELEVENLABS = "elevenlabs"


class AudioModelCreateParams(BaseModel):
    org_id: str = Field(
        description="The organization id of the audio model",
    )
    name: str = Field(
        description="The name of the audio model",
    )
    provider: AudioModelProvider = Field(
        description="The provider of the audio model",
    )
    model_name: str = Field(
        description="The model name of the audio model",
    )
    model_id: str = Field(
        description="The model id of the audio model",
    )
    description: str = Field(
        description="The description of the audio model",
    )
    model_metadata: Optional[Dict[str, Any]] = Field(
        default={},
        description="The model metadata of the audio model",
    )


class AudioModelResponseParams(BaseModel):
    id: uuid.UUID = Field(
        description="The id of the audio model",
    )
    org_id: str = Field(
        description="The organization id of the audio model",
    )
    name: str = Field(
        description="The name of the audio model",
    )
    provider: AudioModelProvider = Field(
        description="The provider of the audio model",
    )
    model_name: str = Field(
        description="The model name of the audio model",
    )
    model_id: str = Field(
        description="The model id of the audio model",
    )
    model_metadata: Optional[Dict[str, Any]] = Field(
        default={},
        description="The model metadata of the audio model",
    )
    description: str = Field(
        description="The description of the audio model",
    )
    created_at: datetime.datetime = Field(
        description="The created at of the audio model",
    )
    updated_at: datetime.datetime = Field(
        description="The updated at of the audio model",
    )
