# processor.py
import json
from typing import List
from llm_sdk import Small_LLM_Model
from src.models.models import FunctionDefinition, PromptInput
from src.engine.generator import generate_function_call


def _build_system_prompt(functions: List[FunctionDefinition]) -> str:
    lines = ["You are a function calling engine. Available functions:"]
    for fn in functions:
        lines.append(f"- {fn.name}: {fn.description}, parameters: {fn.parameters}")
    lines.append("\nCRITICAL: Match the action/verb in the prompt to the right function.")
    return "\n".join(lines)


def process_prompts(
    prompts: List[PromptInput],
    functions: List[FunctionDefinition],
    model: Small_LLM_Model,
) -> str:
    with open(model.get_path_to_vocab_file(), "r", encoding="utf-8") as f:
        vocab = json.load(f)

    system_prompt = _build_system_prompt(functions)
    results = []

    for prompt in prompts:
        try:
            call = generate_function_call(prompt, functions, model, vocab, system_prompt)
            results.append({"prompt": call.prompt, "fn_name": call.fn_name, "args": call.args})
        except Exception as e:
            results.append({"prompt": prompt.prompt, "error": str(e)})

    return json.dumps(results, indent=2)