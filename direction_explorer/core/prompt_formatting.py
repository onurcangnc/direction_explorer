"""Chat-template formatting + response-text extraction."""

from __future__ import annotations

from direction_explorer.core.model_context import ModelContext


def format_chat_text(tokenizer, user_msg: str) -> str:
    """Wrap user message via the model's chat template (text only)."""
    messages = [{"role": "user", "content": user_msg}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return user_msg


def format_prompt(ctx: ModelContext, user_msg: str):
    """Return tokenized inputs for a chat-templated user message."""
    text = format_chat_text(ctx.tokenizer, user_msg)
    return ctx.tokenizer(text, return_tensors="pt").to(ctx.device)


def extract_response_text(generated_text: str, formatted_prompt: str) -> str:
    """Strip the chat-templated prompt from generation output. Tries prefix
    strip first, falls back to common end-of-instruction markers (Llama
    [INST]…[/INST], Qwen <|im_start|>assistant…)."""
    if not generated_text:
        return ""
    if generated_text.startswith(formatted_prompt):
        rest = generated_text[len(formatted_prompt):]
    else:
        rest = generated_text
        for marker in ("[/INST]", "<|im_start|>assistant\n", "\nassistant\n"):
            idx = rest.rfind(marker)
            if idx != -1:
                rest = rest[idx + len(marker):]
                break
    for end in ("</s>", "<|im_end|>", "<|endoftext|>"):
        end_idx = rest.find(end)
        if end_idx != -1:
            rest = rest[:end_idx]
    return rest.strip()
