"""Document and query generation for training data."""

from __future__ import annotations


def tokenize_doc_query(
    tokenizer,
    doc: str,
    question: str,
    options: list[str],
    max_doc_tokens: int,
) -> tuple[list[int], list[int]]:
    """Tokenize a document and query for model input.
    
    Args:
        tokenizer: Hugging Face tokenizer instance.
        doc: The document text.
        question: The question text.
        options: List of answer options.
        max_doc_tokens: Maximum number of tokens for the document.
    
    Returns:
        Tuple of (doc_token_ids, query_token_ids).
    """
    # Tokenize the document and truncate to max_doc_tokens
    doc_tokens = tokenizer.encode(doc, add_special_tokens=False)
    doc_tokens = doc_tokens[:max_doc_tokens]
    
    # Format and tokenize the query
    from still.data.quality import format_query
    query_text = format_query(question, options)
    query_tokens = tokenizer.encode(query_text, add_special_tokens=False)
    
    return doc_tokens, query_tokens
