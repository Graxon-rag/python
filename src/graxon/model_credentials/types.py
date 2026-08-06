from pydantic import BaseModel, Field
from ..types import ModelProvider
import datetime
import uuid


class ModelCredentialCreateParams(BaseModel):
    org_id: str = Field(
        description="The organization id of the model credential",
    )
    name: str = Field(
        description="The name of the model credential",
    )
    description: str = Field(
        description="The description of the model credential",
    )
    provider: ModelProvider = Field(
        description="The provider of the model credential",
    )
    api_key: str = Field(
        description="The api key of the model credential",
    )


class ModelCredentialResponseParams(BaseModel):
    id: uuid.UUID = Field(
        description="The id of the model credential",
    )
    org_id: str = Field(
        description="The organization id of the model credential",
    )
    name: str = Field(
        description="The name of the model credential",
    )
    description: str = Field(
        description="The description of the model credential",
    )
    provider: ModelProvider = Field(
        description="The provider of the model credential",
    )
    api_key: str = Field(
        description="The api key of the model credential",
    )
    created_at: datetime.datetime = Field(
        description="The created at of the model credential",
    )
    updated_at: datetime.datetime = Field(
        description="The updated at of the model credential",
    )
