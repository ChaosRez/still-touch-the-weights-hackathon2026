from __future__ import annotations

import json

import pytest
import torch

from still.config import STILLConfig
from still.data.recurrent_synthetic import generate_evaluation_rows
from still.eval_recurrent import (
    evaluate_recurrent_comparison,
    write_comparison_artifacts,
)
from still.model.wrapper import STILLModel


def _checkpoint(model, path, stage):
    torch.save(
        {
            "format": "still-recurrent-v1",
            "stage": stage,
            "completed_steps": 1,
            "config": {
                "num_latents": model.cfg.num_latents,
                "latent_dim": model.cfg.latent_dim,
                "num_blocks": model.cfg.num_blocks,
            },
            "perceiver": model.perceiver.state_dict(),
        },
        path,
    )


def test_recurrent_comparison_evaluates_four_equal_budget_arms(
    tmp_path, tiny_model_path, tokenizer
):
    cfg = STILLConfig(
        model_name=tiny_model_path,
        num_latents=4,
        latent_dim=16,
        num_blocks=2,
        device="cpu",
    )
    model = STILLModel(tiny_model_path, cfg=cfg, device="cpu")
    single = tmp_path / "single.pt"
    recurrent = tmp_path / "recurrent.pt"
    _checkpoint(model, single, "single_step")
    _checkpoint(model, recurrent, "recurrence_aware")
    rows = generate_evaluation_rows(
        tokenizer,
        per_depth=4,
        seed=99,
        depths=(1, 2),
        chunk_tokens=32,
    )

    result = evaluate_recurrent_comparison(
        model,
        tokenizer,
        rows,
        single_checkpoint=str(single),
        recurrent_checkpoint=str(recurrent),
        memory_positions=4,
    )

    assert set(result["summary"]) == {
        "full_context",
        "text_window",
        "single_step",
        "recurrent",
    }
    assert len(result["records"]) == len(rows) * 4
    for method in ("text_window", "single_step", "recurrent"):
        for depth in (1, 2):
            metrics = result["summary"][method]["by_depth"][str(depth)]["overall"]
            assert metrics["memory_positions"] == 4
            assert metrics["source_tokens_per_position"] == depth * 8
    assert result["summary"]["text_window"]["by_depth"]["2"]["overall"][
        "represented_source_tokens"
    ] == 4
    assert result["summary"]["recurrent"]["by_depth"]["2"]["overall"][
        "represented_source_tokens"
    ] == 64
    assert result["summary"]["recurrent"]["by_depth"]["2"]["overall"][
        "mean_compaction_ms_per_chunk"
    ] >= 0


def test_artifact_writer_preserves_existing_plot_runs(tmp_path):
    depth_metrics = {
        "overall": {
            "n": 1,
            "accuracy": 1.0,
            "mean_answer_ce": 0.1,
            "mean_teacher_answer_ce": 0.1,
            "mean_ce_gap_to_full": 0.0,
            "mean_kl_to_full": 0.0,
            "source_tokens": 64,
            "represented_source_tokens": 64,
            "memory_positions": 64,
            "source_tokens_per_position": 1.0,
            "mean_compaction_ms_per_chunk": None,
        },
        "oldest": {
            "n": 1,
            "accuracy": 1.0,
            "mean_answer_ce": 0.1,
            "mean_teacher_answer_ce": 0.1,
            "mean_ce_gap_to_full": 0.0,
            "mean_kl_to_full": 0.0,
            "source_tokens": 64,
            "represented_source_tokens": 64,
            "memory_positions": 64,
            "source_tokens_per_position": 1.0,
            "mean_compaction_ms_per_chunk": None,
        },
    }
    result = {
        "summary": {
            method: {"label": method, "by_depth": {"1": depth_metrics}}
            for method in ("full_context", "text_window", "single_step", "recurrent")
        },
        "records": [],
    }

    write_comparison_artifacts(
        result, output_root=str(tmp_path), run_name="run-v1", depths=(1,)
    )

    assert (tmp_path / "plots/run-v1/oldest_accuracy_vs_depth.png").exists()
    assert json.loads((tmp_path / "metrics/run-v1/comparison.json").read_text()) == result
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_comparison_artifacts(
            result, output_root=str(tmp_path), run_name="run-v1", depths=(1,)
        )
