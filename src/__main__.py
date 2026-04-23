import argparse
from llm_sdk import Small_LLM_Model
from src.parser.parser import load_prompts, load_definitions
from src.engine.processor import process_prompts
from src.engine.generate_output_file import generate_output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Function Calling Engine")
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file"
    )
    parser.add_argument(
        "--functions",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the functions definition file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to the output JSON file"
    )
    args = parser.parse_args()

    prompts = load_prompts(args.input)
    functions = load_definitions(args.functions)
    model = Small_LLM_Model()
    result = process_prompts(prompts, functions, model)
    generate_output_file(result, args.output)


if __name__ == "__main__":
    main()