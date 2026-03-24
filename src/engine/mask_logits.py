import numpy as np


def mask_fn_name_logits(
    logits: np.ndarray,
    generated: str,
    valid_names: list[str],
    vocab: dict,
) -> np.ndarray:
    """
    Only allow tokens that keep matching a valid function name
    """
    valid_tokens = set()

    for token_id, token_str in vocab.items():
        for name in valid_names:
            if (generated + token_str).startswith(name):
                valid_tokens.add(int(token_id))

    mask = np.full_like(logits, -np.inf)
    for idx in valid_tokens:
        mask[idx] = logits[idx]

    return mask


def mask_params_logits(
    logits: np.ndarray,
    generated: str,
    param_type: str,
    vocab: dict,
) -> np.ndarray:
    """
    Restrict tokens depending on type
    """
    valid_tokens = set()

    for token_id, token_str in vocab.items():
        if param_type == "number":
            if token_str.strip().replace(".", "").isdigit() or token_str in [",", "}"]:
                valid_tokens.add(int(token_id))

        elif param_type == "string":
            valid_tokens.add(int(token_id))

    mask = np.full_like(logits, -np.inf)
    for idx in valid_tokens:
        mask[idx] = logits[idx]

    return mask