from __future__ import annotations

from collections import Counter

from still.data.recurrent_synthetic import (
    EVAL_DEPTHS,
    TRAIN_DEPTHS,
    generate_evaluation_rows,
    generate_recurrent_rows,
)


def test_recurrent_rows_are_deterministic_unique_and_fixed_length(tokenizer):
    kwargs = {
        "count": 16,
        "depths": TRAIN_DEPTHS,
        "seed": 11,
        "chunk_tokens": 48,
    }
    first = generate_recurrent_rows(tokenizer, **kwargs)
    second = generate_recurrent_rows(tokenizer, **kwargs)

    assert first == second
    assert Counter(row["target_age"] for row in first) == {
        "oldest": 8,
        "middle": 4,
        "newest": 4,
    }
    for row in first:
        assert row["depth"] in TRAIN_DEPTHS
        assert len(row["chunks_input_ids"]) == row["depth"]
        assert all(len(chunk) == 48 for chunk in row["chunks_input_ids"])
        assert len(set(row["projects"])) == row["depth"]
        assert len(set(row["values"])) == row["depth"]
        assert len(set(row["options"])) == 4
        assert row["options"][row["correct_index"]] == row["values"][row["target_chunk"]]
        assert row["answer_input_ids"]


def test_evaluation_rows_cover_every_requested_depth(tokenizer):
    rows = generate_evaluation_rows(tokenizer, per_depth=4, seed=29, chunk_tokens=48)

    assert Counter(row["depth"] for row in rows) == {depth: 4 for depth in EVAL_DEPTHS}
    for depth in EVAL_DEPTHS:
        ages = Counter(row["target_age"] for row in rows if row["depth"] == depth)
        assert ages == {"oldest": 2, "middle": 1, "newest": 1}


def test_weighted_depths_are_stratified_to_the_requested_mix(tokenizer):
    rows = generate_recurrent_rows(
        tokenizer,
        count=75,
        depths=(1, 2, 4, 8),
        depth_weights=(0.1, 0.2, 0.3, 0.4),
        seed=17,
        chunk_tokens=48,
    )

    assert Counter(row["depth"] for row in rows) == {1: 8, 2: 15, 4: 22, 8: 30}
