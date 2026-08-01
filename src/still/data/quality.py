"""Quality metrics and utilities for evaluating STILL models."""

from __future__ import annotations

ANSWER_CUE = "Answer:"
LETTERS = ["A", "B", "C", "D"]


def format_query(question: str, options: list[str]) -> str:
    """Format a question with multiple-choice options for LLM input.
    
    Args:
        question: The question text.
        options: List of answer options (exactly 4).
    
    Returns:
        Formatted query string with question and options.
    """
    formatted_options = "\n".join(f"{letter}. {option}" for letter, option in zip(LETTERS, options))
    return f"{question}\n{formatted_options}\n{ANSWER_CUE}"


def letter_token_ids(tokenizer) -> list[int]:
    """Get token IDs for the answer letters (A, B, C, D).
    
    Args:
        tokenizer: Hugging Face tokenizer instance.
    
    Returns:
        List of 4 token IDs corresponding to A, B, C, D.
    """
    token_ids = []
    for letter in LETTERS:
        # Tokenize just the letter and get the first token
        tokens = tokenizer.encode(letter, add_special_tokens=False)
        token_ids.append(tokens[0] if tokens else tokenizer.unk_token_id)
    return token_ids
