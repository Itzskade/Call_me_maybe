from typing import Dict, Any

try:
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "[ERROR] Pydantic not installed"
        "Usage: pip install pydantic"
        )

class Parameter(BaseModel):
    type: str

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Parameter

class PromptInput(BaseModel):
    prompt: str

class FunctionCallOutput(BaseModel):
    prompt: str
    fn_name: str
    args: Dict[str, Any]