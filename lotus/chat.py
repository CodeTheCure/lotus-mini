def build_messages(row: dict) -> dict:
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
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": query}], add_generation_prompt=True
    )
