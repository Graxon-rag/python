from ..project_configs.types import ProjectConfigCreateParams
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import datetime
import uuid


class ProjectCreateParams(BaseModel):
    org_id: str = Field(
        description="The organization id of the project",
    )
    name: str = Field(
        description="The name of the project",
    )
    config: ProjectConfigCreateParams
    description: str = Field(
        description="The description of the project",
    )
    project_metadata: Optional[Dict[str, Any]] = Field(
        default={},
        description="The project metadata of the project",
    )


class ProjectResponseParams(BaseModel):
    id: uuid.UUID = Field(
        description="The id of the project",
    )
    readable_id: str = Field(
        description="The readable id of the project",
    )
    org_id: str = Field(
        description="The organization id of the project",
    )
    name: str = Field(
        description="The name of the project",
    )
    description: str = Field(
        description="The description of the project",
    )
    created_at: datetime.datetime = Field(
        description="The created at of the project",
    )
    updated_at: datetime.datetime = Field(
        description="The updated at of the project",
    )
