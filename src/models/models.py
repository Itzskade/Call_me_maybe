from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict


class ParamType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"


class Parameter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ParamType


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter


class PromptInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str


class FunctionCallOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    fn_name: str
    args: dict[str, Any]