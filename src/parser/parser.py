import json
import sys
from typing import List
from pydantic import ValidationError
from src.schemas.schemas import FunctionDefinition, TestPrompt


def _load_json_list(file_path: str, model) -> list:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"Error: {file_path} must contain a list.", file=sys.stderr)
            sys.exit(1)
        return [model(**item) for item in data]
    except OSError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: {file_path} is not a valid JSON file.", file=sys.stderr)
        sys.exit(1)
    except ValidationError as e:
        print(f"Validation error in {file_path}:\n{e}", file=sys.stderr)
        sys.exit(1)


def load_prompts(file_path: str) -> List[TestPrompt]:
    """Load and validate the prompts test file."""
    return _load_json_list(file_path, TestPrompt)


def load_definitions(file_path: str) -> List[FunctionDefinition]:
    """Load and validate the function definitions file."""
    return _load_json_list(file_path, FunctionDefinition)