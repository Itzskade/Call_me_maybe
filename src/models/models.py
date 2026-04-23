from typing import Dict, Any

try:
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "[ERROR] Pydantic not installed"
        "Usage: pip install pydantic"
        )


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
