import json
import sys
from typing import List
from pydantic import ValidationError
from src.models.models import FunctionDefinition, PromptInput


def _load_json_list(file_path: str) -> list[dict]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"Error: {file_path} must contain a list.", file=sys.stderr)
            sys.exit(1)

        if not all(isinstance(x, dict) for x in data):
            print(f"Error: {file_path} must contain a list of objects.", file=sys.stderr)
            sys.exit(1)

        return data

    except OSError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        sys.exit(1)

    except json.JSONDecodeError:
        print(f"Error: {file_path} is not a valid JSON file.", file=sys.stderr)
        sys.exit(1)


def load_prompts(file_path: str) -> List[PromptInput]:
    """Load and validate prompts."""
    raw_data = _load_json_list(file_path)

    prompts: List[PromptInput] = []

    for i, item in enumerate(raw_data):
        try:
            prompts.append(PromptInput.model_validate(item))
        except ValidationError as e:
            print(f"[SKIP] invalid prompt at index {i}:\n{e}", file=sys.stderr)

    return prompts


def load_definitions(file_path: str) -> List[FunctionDefinition]:
    """Load and validate function definitions."""
    raw_data = _load_json_list(file_path)

    functions: List[FunctionDefinition] = []

    for i, item in enumerate(raw_data):
        try:
            functions.append(FunctionDefinition.model_validate(item))
        except ValidationError as e:
            print(f"[SKIP] invalid function at index {i}:\n{e}", file=sys.stderr)

    return functions
