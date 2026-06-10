*This project has been created as part of the 42 curriculum by <rmarin-n>.*

# Call Me Maybe

## Description

**Call Me Maybe** is a function-calling system based on **constrained decoding** over a local language model.

The goal is to map natural language prompts to predefined functions and their arguments while ensuring the output remains structured and valid by controlling token generation through logit masking.

The system:
- Selects the most appropriate function for a given prompt.
- Generates function arguments in a controlled manner.
- Prevents invalid outputs through vocabulary-level constraints.

---

## Instructions

### Installation
```
uv sync
```
### Execution
```
uv run python -m src
```
With custom paths:
```
uv run python -m src --input data/input/function_calling_tests.json \
                     --functions data/input/functions_definition.json \
                     --output data/output/results.json
```
### Lint
```
make lint
```
---

## Algorithm (Constrained Decoding)

The model generates tokens step by step while applying constraints at each iteration to restrict valid outputs.

- **Function selection**: only tokens matching valid function name prefixes are allowed.
- **Numeric parameters**: restricted to digits, decimal point, and sign.
- **String parameters**: generated freely until a termination condition is reached.

Final selection is performed using argmax over masked logits.

---

## Design Decisions

- Two-stage pipeline: function selection → argument generation.
- External logit masking instead of fine-tuning.
- Input validation using Pydantic models.
- Fully local inference using Hugging Face Transformers.

---

## Performance

- **Accuracy**: high for function selection due to strict constraints.
- **Speed**: limited by token-by-token inference.
- **Reliability**: prevents structurally invalid outputs, though semantic accuracy may vary on ambiguous prompts.

---

## Challenges

- Tokenization does not always align with characters.
- Detecting string termination under constrained decoding.
- Efficiently building and applying vocabulary masks.

---

## Testing Strategy

Tests are defined in `data/input/function_calling_tests.json` and cover:

- arithmetic operations
- string transformations
- regex-style substitutions

Validation checks:
- selected function correctness
- output format validity
- type consistency

---

## Resources

- https://huggingface.co/docs/transformers
- https://docs.pydantic.dev
- Constrained decoding and structured generation literature

**AI usage**: used for documentation drafting, algorithm explanation, and design review support.
