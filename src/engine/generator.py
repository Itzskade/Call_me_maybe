import json
import numpy as np
from typing import List
from llm_sdk import Small_LLM_Model
from models import FunctionDefinition, PromptInput, FunctionCallOutput
from mask_logits import mask_fn_name_logits, mask_params_logits


def generate_function_call(
    prompt: PromptInput,
    functions: List[FunctionDefinition],
    model: Small_LLM_Model
) -> FunctionCallOutput:
    """
    Generate function call using constrained decoding (partial).
    """
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    fn_names = [fn.name for fn in functions]
    fn_map = {fn.name: fn for fn in functions}

    generated = ""

    while True:
        text = f"{prompt.prompt}\nFunction: \"{generated}"
        input_ids = model.encode(text)

        logits = model.get_logits_from_input_ids(input_ids[0].tolist())
        logits = np.array(logits)

        logits = mask_fn_name_logits(logits, generated, fn_names, vocab)

        next_token_id = int(np.argmax(logits))
        next_token = model.decode([next_token_id])

        generated += next_token

        if next_token == '"':
            break

    fn_name = generated[:-1]

    args = {}
    current_fn = fn_map[fn_name]

    for param_name, param in current_fn.parameters.items():

        generated_param = ""

        while True:
            text = f"{prompt.prompt}\n{fn_name}({param_name}={generated_param}"
            input_ids = model.encode(text)

            logits = model.get_logits_from_input_ids(input_ids[0].tolist())
            logits = np.array(logits)

            logits = mask_params_logits(
                logits,
                generated_param,
                param.type,
                vocab
            )

            next_token_id = int(np.argmax(logits))
            next_token = model.decode([next_token_id])

            generated_param += next_token

            if param.type == "string" and '"' in generated_param:
                value = generated_param.split('"')[0]
                args[param_name] = value
                break

            if param.type == "number" and (
                next_token in [",", "}"]
            ):
                try:
                    value = float(generated_param.strip(",} "))
                except:
                    value = 0.0
                args[param_name] = value
                break

    return FunctionCallOutput(
        prompt=prompt.prompt,
        fn_name=fn_name,
        args=args
    )