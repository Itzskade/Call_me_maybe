# generator.py
import numpy as np
from typing import List
from llm_sdk import Small_LLM_Model
from src.models.models import FunctionDefinition, PromptInput, FunctionCallOutput
from src.engine.mask_logits import mask_fn_name_logits, mask_params_logits


def _generate_tokens(
    context: str,
    model: Small_LLM_Model,
    logits_mask_fn,
    mask_args: tuple,
    max_tokens: int = 100,
) -> str:
    generated = ""
    for _ in range(max_tokens):
        input_ids = model.encode(context + generated)
        logits = np.array(model.get_logits_from_input_ids(input_ids[0].tolist()))
        logits = logits_mask_fn(logits, generated, *mask_args)
        token = model.decode([int(np.argmax(logits))])
        generated += token
        if token in ('"', ",", "}"):
            break
    else:
        raise RuntimeError(f"Max tokens reached. Generated: '{generated}'")
    return generated


def generate_function_call(
    prompt: PromptInput,
    functions: List[FunctionDefinition],
    model: Small_LLM_Model,
    vocab: dict,
    system_prompt: str,
) -> FunctionCallOutput:
    fn_names = [fn.name for fn in functions]
    fn_map = {fn.name: fn for fn in functions}

    fn_name = _generate_tokens(
        context=f"{system_prompt}\n{prompt.prompt}\nFunction: \"",
        model=model,
        logits_mask_fn=mask_fn_name_logits,
        mask_args=(fn_names, vocab),
    )[:-1]

    if fn_name not in fn_map:
        raise KeyError(f"Unknown function '{fn_name}'. Available: {fn_names}")

    args = {}
    for param_name, param in fn_map[fn_name].parameters.items():
        prefix = '"' if param.type == "string" else ""
        raw = _generate_tokens(
            context=f"{system_prompt}\n{prompt.prompt}\n{fn_name}({param_name}={prefix}",
            model=model,
            logits_mask_fn=mask_params_logits,
            mask_args=(None, param.type, vocab),
        )
        if param.type == "number":
            try:
                args[param_name] = float(raw.rstrip(',"} '))
            except ValueError:
                args[param_name] = 0.0
        else:
            args[param_name] = raw[:-1]

    return FunctionCallOutput(prompt=prompt.prompt, fn_name=fn_name, args=args)