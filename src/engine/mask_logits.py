import numpy as np


def mask_fn_name_logits(
    logits: np.ndarray,
    generated: str,
    valid_names: list[str],
    vocab: dict[str, int],
) -> np.ndarray:
    valid_tokens: set[int] = set()

    is_complete = generated in valid_names

    for token_str, token_id in vocab.items():
        if is_complete:
            if token_str == '"':
                valid_tokens.add(token_id)
            continue

        candidate = generated + token_str

        for name in valid_names:
            if name.startswith(candidate):
                valid_tokens.add(token_id)
                break

    if not valid_tokens:
        return logits

    mask = np.full_like(logits, -np.inf)

    for idx in valid_tokens:
        if 0 <= idx < len(logits):
            mask[idx] = logits[idx]

    return mask


def _is_valid_number_token(token: str, generated: str) -> bool:
    t = token.strip()

    if not t:
        return False

    if t == "-":
        return len(generated.strip()) == 0

    if t == ".":
        return "." not in generated

    return t.isdigit()


def mask_params_logits(
    logits: np.ndarray,
    generated: str,
    param_type: str,
    vocab: dict[str, int],
) -> np.ndarray:
    valid_tokens: set[int] = set()

    for token_str, token_id in vocab.items():

        if param_type == "number":
            if token_str in (",", "}"):
                valid_tokens.add(token_id)
                continue

            if _is_valid_number_token(token_str, generated):
                valid_tokens.add(token_id)

        elif param_type == "string":
            if any(x in generated for x in ['"', "\n", "CRITICAL"]):
                if token_str in ('"', ",", "}"):
                    valid_tokens.add(token_id)
                continue

            valid_tokens.add(token_id)

    if not valid_tokens:
        return logits

    mask = np.full_like(logits, -np.inf)

    for idx in valid_tokens:
        if 0 <= idx < len(logits):
            mask[idx] = logits[idx]

    return mask