import json
import os
from typing import Any


def generate_output_file(content: list[dict], output_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)

    except OSError as e:
        print(f"Error writing output file: {e}")