import os
from src.parser.parser import load_prompts, load_functions
from src.engine.processor import process_prompts
from llm_sdk import Small_LLM_Model


def main():
    prompts_path = "data/input/function_calling_tests.json"
    functions_path = "data/input/functions_definition.json"
    output_path = "data/output/output.json"

    prompts = load_prompts(prompts_path)
    functions = load_functions(functions_path)

    model = Small_LLM_Model()
    result_json = process_prompts(prompts, functions, model)

    os.makedirs("data/output", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result_json)

    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    main()
