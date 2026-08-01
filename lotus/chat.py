"""Chat formatting helpers shared between data-prep and inference."""


def build_messages(row: dict) -> dict:
    """Convert a raw medical-QA row into mlx-lm chat `{"messages": [...]}` format."""
    instruction = row.get("instruction") or ""
    user_input = row.get("input") or ""
    output = row.get("output") or ""
    prompt = f"{instruction}\n\nPatient: {user_input}".strip() if user_input else instruction
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": output},
        ]
    }


def apply_chat_template(tokenizer, query: str) -> str:
    """Wrap a user query in the model's chat template, ready for generation."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": query}], add_generation_prompt=True
    )
