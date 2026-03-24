import json
from typing import List
from engine.generator import generate_function_call
from models.models import PromptInput, FunctionDefinition


def process_prompts(prompts: List[PromptInput], functions: List[FunctionDefinition], model) -> str:
    results = []
    for prompt in prompts:
        call = generate_function_call(prompt, functions, model)
        results.append({
            "prompt": call.prompt,
            "fn_name": call.fn_name,
            "args": call.args
        })
    return json.dumps(results, indent=2)