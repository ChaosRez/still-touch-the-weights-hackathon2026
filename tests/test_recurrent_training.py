from __future__ import annotations

import os

import torch

from still.config import STILLConfig
from still.data.recurrent_synthetic import generate_recurrent_rows
from still.model.wrapper import STILLModel
from still.train_recurrent import build_student_cache, run_training_stage


def _model(tiny_model_path, num_latents=4):
    cfg = STILLConfig(
        model_name=tiny_model_path,
        num_latents=num_latents,
        latent_dim=16,
        num_blocks=2,
        device="cpu",
    )
    return STILLModel(tiny_model_path, cfg=cfg, device="cpu")


def test_ten_training_recurrences_stay_fixed_finite_and_differentiable(tiny_model_path):
    model = _model(tiny_model_path, num_latents=64)
    state = model.compact_tokens(torch.randint(0, 100, (16,)).tolist())

    for _ in range(10):
        state = model.recompact_train(
            state,
            torch.randint(0, 100, (12,)).tolist(),
            detach_prior=True,
        )
        assert state.num_latents == 64
        assert all(torch.isfinite(tensor).all() for tensor in state.compact_k)
        assert all(torch.isfinite(tensor).all() for tensor in state.compact_v)

    logits = model.decode([1, 2, 3], [4], state)
    loss = logits.float().square().mean()
    loss.backward()
    gradient = sum(
        parameter.grad.detach().float().square().sum()
        for parameter in model.perceiver.parameters()
        if parameter.grad is not None
    ).sqrt()
    assert torch.isfinite(loss)
    assert gradient > 0
    assert all(parameter.grad is None for parameter in model.base.parameters())


def test_single_and_recurrent_stage_write_resumable_checkpoints(
    tmp_path, tiny_model_path, tokenizer
):
    model = _model(tiny_model_path)
    single_rows = generate_recurrent_rows(
        tokenizer, count=1, depths=(1,), seed=3, chunk_tokens=48
    )
    single_path = str(tmp_path / "single.pt")
    single = run_training_stage(
        model,
        single_rows,
        stage="single_step",
        steps=1,
        learning_rate=1e-3,
        output_path=single_path,
    )

    recurrent_rows = generate_recurrent_rows(
        tokenizer, count=1, depths=(2,), seed=4, chunk_tokens=48
    )
    recurrent_path = str(tmp_path / "recurrent.pt")
    recurrent = run_training_stage(
        model,
        recurrent_rows,
        stage="recurrence_aware",
        steps=1,
        learning_rate=1e-3,
        output_path=recurrent_path,
        resume_path=single_path,
    )

    assert os.path.exists(single["checkpoint"])
    assert os.path.exists(recurrent["checkpoint"])
    assert single["logs"][0]["depth"] == 1
    assert recurrent["logs"][0]["depth"] == 2
    assert single["logs"][0]["perceiver_grad_norm"] > 0
    assert recurrent["logs"][0]["perceiver_grad_norm"] > 0


def test_build_student_cache_has_exact_configured_budget(tiny_model_path):
    model = _model(tiny_model_path)
    chunks = [torch.randint(0, 100, (12,)).tolist() for _ in range(4)]

    cache = build_student_cache(model, chunks)

    assert cache.num_latents == model.cfg.num_latents
