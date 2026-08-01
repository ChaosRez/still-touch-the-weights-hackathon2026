"""Deterministic synthetic facts for repeated fixed-budget compaction.

Rows are deliberately simple and auditable: each short chunk assigns one access code
to one unique project, and the query asks for one project through a four-way MCQ.  No
Alien API prompts, labels, feedback, or metadata enter this dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

TargetAge = Literal["oldest", "middle", "newest"]
TRAIN_DEPTHS = (1, 2, 4, 8)
EVAL_DEPTHS = (1, 2, 4, 8, 16, 32)
LETTERS = ("A", "B", "C", "D")


@dataclass(frozen=True, slots=True)
class RecurrentSyntheticConfig:
    chunk_tokens: int = 64
    train_depths: tuple[int, ...] = TRAIN_DEPTHS
    eval_depths: tuple[int, ...] = EVAL_DEPTHS
    train_seed: int = 1701
    eval_seed: int = 2903


def target_age_for_row(index: int) -> TargetAge:
    """Give each group of four rows an exact 50/25/25 age split."""

    return ("oldest", "oldest", "middle", "newest")[index % 4]


def target_index_for_age(depth: int, target_age: TargetAge) -> int:
    if depth < 1:
        raise ValueError("depth must be positive")
    if target_age == "oldest":
        return 0
    if target_age == "newest":
        return depth - 1
    if target_age == "middle":
        return depth // 2
    raise ValueError(f"unknown target age: {target_age}")


def _encode(tokenizer, text: str) -> list[int]:
    return [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]


def _fixed_chunk(tokenizer, fact: str, chunk_tokens: int, chunk_index: int) -> list[int]:
    fact_ids = _encode(tokenizer, fact)
    if len(fact_ids) > chunk_tokens:
        raise ValueError(
            f"chunk_tokens={chunk_tokens} is too small for the {len(fact_ids)}-token fact"
        )
    filler = (
        f" Audit record {chunk_index} was checked twice and contains no other access codes."
    )
    text = fact
    ids = fact_ids
    while len(ids) < chunk_tokens:
        text += filler
        ids = _encode(tokenizer, text)
    return ids[:chunk_tokens]


def _one_token_answer(tokenizer, letter: str) -> list[int]:
    # A leading space is the natural continuation after ``Answer:`` for Qwen.
    spaced = _encode(tokenizer, f" {letter}")
    if len(spaced) == 1:
        return spaced
    plain = _encode(tokenizer, letter)
    return plain or spaced


def generate_recurrent_example(
    tokenizer,
    *,
    depth: int,
    target_age: TargetAge,
    seed: int,
    example_index: int,
    chunk_tokens: int = 64,
) -> dict:
    """Generate one tokenized repeated-compaction example."""

    rng = random.Random((seed << 20) ^ example_index ^ (depth << 8))
    target_chunk = target_index_for_age(depth, target_age)
    projects = [
        f"P{example_index:05d}-{chunk_index:03d}-{rng.randrange(1000, 9999)}"
        for chunk_index in range(depth)
    ]
    values = rng.sample(range(1000, 10000), depth)
    chunks_text = [
        f"The access code for Project {project} is {value}."
        for project, value in zip(projects, values, strict=True)
    ]
    chunks_input_ids = [
        _fixed_chunk(tokenizer, text, chunk_tokens, chunk_index)
        for chunk_index, text in enumerate(chunks_text)
    ]

    correct_value = values[target_chunk]
    distractor_pool = [value for i, value in enumerate(values) if i != target_chunk]
    while len(distractor_pool) < 3:
        candidate = rng.randrange(1000, 10000)
        if candidate != correct_value and candidate not in distractor_pool:
            distractor_pool.append(candidate)
    options = [correct_value, *rng.sample(distractor_pool, 3)]
    rng.shuffle(options)
    correct_index = options.index(correct_value)
    question = f"What is the access code for Project {projects[target_chunk]}?"
    rendered_options = "\n".join(
        f"{letter}. {value}" for letter, value in zip(LETTERS, options, strict=True)
    )
    query = f"{question}\n{rendered_options}\nAnswer:"
    answer_letter = LETTERS[correct_index]

    return {
        "chunks_input_ids": chunks_input_ids,
        "query_input_ids": _encode(tokenizer, query),
        "answer_input_ids": _one_token_answer(tokenizer, answer_letter),
        "depth": depth,
        "target_chunk": target_chunk,
        "target_age": target_age,
        "correct_index": correct_index,
        "answer_letter": answer_letter,
        "projects": projects,
        "values": values,
        "options": options,
        "chunks_text": chunks_text,
        "question": question,
    }


def generate_recurrent_rows(
    tokenizer,
    *,
    count: int,
    depths: Sequence[int],
    seed: int,
    chunk_tokens: int = 64,
    depth_weights: Sequence[float] | None = None,
) -> list[dict]:
    """Generate a deterministic row list with an exact repeating age distribution."""

    if count < 0:
        raise ValueError("count must be non-negative")
    if not depths or any(depth < 1 for depth in depths):
        raise ValueError("depths must contain positive integers")
    if depth_weights is not None and len(depth_weights) != len(depths):
        raise ValueError("depth_weights must match depths")
    rng = random.Random(seed)
    if depth_weights is None:
        depth_schedule = [rng.choice(list(depths)) for _ in range(count)]
    else:
        if any(weight < 0 for weight in depth_weights) or sum(depth_weights) <= 0:
            raise ValueError("depth_weights must be non-negative with a positive sum")
        total_weight = sum(depth_weights)
        exact_counts = [count * weight / total_weight for weight in depth_weights]
        counts = [math.floor(value) for value in exact_counts]
        remainder = count - sum(counts)
        fractional_order = sorted(
            range(len(depths)),
            key=lambda i: (exact_counts[i] - counts[i], -i),
            reverse=True,
        )
        for i in fractional_order[:remainder]:
            counts[i] += 1
        depth_schedule = [
            depth
            for depth, depth_count in zip(depths, counts, strict=True)
            for _ in range(depth_count)
        ]
        rng.shuffle(depth_schedule)

    rows = []
    for index, depth in enumerate(depth_schedule):
        rows.append(
            generate_recurrent_example(
                tokenizer,
                depth=depth,
                target_age=target_age_for_row(index),
                seed=seed,
                example_index=index,
                chunk_tokens=chunk_tokens,
            )
        )
    return rows


def generate_evaluation_rows(
    tokenizer,
    *,
    per_depth: int,
    seed: int,
    depths: Sequence[int] = EVAL_DEPTHS,
    chunk_tokens: int = 64,
) -> list[dict]:
    """Generate the same number of deterministic evaluation rows at each depth."""

    rows = []
    for depth_index, depth in enumerate(depths):
        for local_index in range(per_depth):
            global_index = depth_index * per_depth + local_index
            rows.append(
                generate_recurrent_example(
                    tokenizer,
                    depth=depth,
                    target_age=target_age_for_row(local_index),
                    seed=seed,
                    example_index=global_index,
                    chunk_tokens=chunk_tokens,
                )
            )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=RecurrentSyntheticConfig().train_seed)
    parser.add_argument("--chunk-tokens", type=int, default=64)
    parser.add_argument("--depths", type=int, nargs="+", default=list(TRAIN_DEPTHS))
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = generate_recurrent_rows(
        tokenizer,
        count=args.count,
        depths=args.depths,
        seed=args.seed,
        chunk_tokens=args.chunk_tokens,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(json.dumps({"config": asdict(RecurrentSyntheticConfig()), "rows": len(rows)}))


if __name__ == "__main__":
    main()
