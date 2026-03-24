import json
from src.parser.parser import load_prompts, load_functions
from engine.processor import process_prompts

def main():
    prompts_path = "data/prompts.json"
    functions_path = "data/functions.json"

    prompts = load_prompts(prompts_path)
    functions = load_functions(functions_path)

    result_json = process_prompts(prompts, functions)

    with open("output.json", "w", encoding="utf-8") as f:
        f.write(result_json)
    
    print("Output saved to output.json")

if __name__ == "__main__":
    main()