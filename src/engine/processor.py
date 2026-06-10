import json
from llm_sdk import Small_LLM_Model
from src.models.models import FunctionDefinition, PromptInput
from src.engine.generator import generate_function_call


def _build_system_prompt(functions: list[FunctionDefinition]) -> str:
    lines = ["You are a function calling engine. Available functions:"]

    for fn in functions:
        lines.append(f"- {fn.name}: {fn.description}, parameters: {fn.parameters}")

    lines.append("\nCRITICAL: Match the action/verb in the prompt to the right function.")
    return "\n".join(lines)


def _load_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_prompts(
    prompts: list[PromptInput],
    functions: list[FunctionDefinition],
    model: Small_LLM_Model,
) -> list[dict]:

    try:
        vocab = _load_json(model.get_path_to_vocab_file())
    except FileNotFoundError as e:
        raise RuntimeError(f"Could not load vocab file: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid vocab JSON: {e}")

    system_prompt = _build_system_prompt(functions)

    results: list[dict] = []

    for prompt in prompts:
        try:
            call = generate_function_call(
                prompt,
                functions,
                model,
                vocab,
                system_prompt,
            )

            results.append({
                "prompt": call.prompt,
                "fn_name": call.fn_name,
                "args": call.args,
            })

        except Exception as e:
            results.append({
                "prompt": prompt.prompt,
                "error": str(e),
            })

    return results