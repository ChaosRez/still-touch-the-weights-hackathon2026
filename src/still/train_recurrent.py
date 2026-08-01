"""Two-stage fixed-budget training for recurrent STILL compaction.

Stage one trains the Perceiver on raw depth-1 K/V.  Stage two resumes that
checkpoint and trains the last differentiable compaction step on on-policy states
sampled at depths 1/2/4/8 while the frozen Qwen base remains unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from still.config import STILLConfig
from still.data.recurrent_synthetic import (
    RecurrentSyntheticConfig,
    generate_recurrent_rows,
)
from still.model.wrapper import STILLModel
from still.train import kl_teacher_student, perceiver_grad_norm

RECURRENT_DEPTHS = (1, 2, 4, 8)
RECURRENT_WEIGHTS = (0.10, 0.20, 0.30, 0.40)


def build_student_cache(model: STILLModel, chunks: Sequence[Sequence[int]]):
    """Build the stage-appropriate cache with gradients only on the final step."""

    if not chunks:
        raise ValueError("an example must contain at least one chunk")
    if len(chunks) == 1:
        return model.compress(chunks[0])

    with torch.no_grad():
        state = model.compact_tokens(chunks[0])
        for chunk in chunks[1:-1]:
            state = model.recompact(state, chunk)
    return model.recompact_train(state, chunks[-1], detach_prior=True)


def _checkpoint_payload(
    model: STILLModel,
    optimizer: torch.optim.Optimizer,
    *,
    stage: str,
    completed_steps: int,
    logs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format": "still-recurrent-v1",
        "stage": stage,
        "completed_steps": completed_steps,
        "model_name": model.cfg.model_name,
        "config": asdict(model.cfg),
        "perceiver": model.perceiver.state_dict(),
        "optimizer": optimizer.state_dict(),
        "logs": logs,
    }


def load_training_checkpoint(
    model: STILLModel,
    path: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=model.perceiver_device, weights_only=True)
    state = payload.get("perceiver", payload)
    model.perceiver.load_state_dict(state)
    if optimizer is not None and isinstance(payload, dict) and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def run_training_stage(
    model: STILLModel,
    rows: Sequence[dict],
    *,
    stage: str,
    steps: int,
    learning_rate: float,
    output_path: str,
    resume_path: str | None = None,
    log_every: int = 1,
) -> dict[str, Any]:
    """Train one comparison stage and write a resumable checkpoint."""

    if steps < 1:
        raise ValueError("steps must be positive")
    if not rows:
        raise ValueError("rows must not be empty")

    model.perceiver.train()
    optimizer = torch.optim.AdamW(model.perceiver.parameters(), lr=learning_rate)
    if resume_path:
        load_training_checkpoint(model, resume_path, optimizer)

    use_cuda = model.device_str.startswith("cuda") and torch.cuda.is_available()
    logs: list[dict[str, Any]] = []
    for step in range(steps):
        started = time.perf_counter()
        row = rows[step % len(rows)]
        chunks = row["chunks_input_ids"]
        query = row["query_input_ids"]
        answer = row["answer_input_ids"]
        full_document = [token for chunk in chunks for token in chunk]

        teacher_logits = model.teacher_logits(full_document, query, answer)
        cache = build_student_cache(model, chunks)
        student_logits = model.decode(query, answer, cache)
        loss = kl_teacher_student(student_logits, teacher_logits)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite KL at step {step}: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = perceiver_grad_norm(model)
        if not grad_norm > 0:
            raise RuntimeError(f"zero Perceiver gradient at step {step}")
        if any(parameter.grad is not None for parameter in model.base.parameters()):
            raise RuntimeError("frozen base model received a gradient")
        optimizer.step()

        answer_targets = torch.tensor(answer, device=student_logits.device, dtype=torch.long)
        teacher_ce = F.cross_entropy(teacher_logits.float(), answer_targets).item()
        student_ce = F.cross_entropy(student_logits.detach().float(), answer_targets).item()
        entry: dict[str, Any] = {
            "step": step + 1,
            "stage": stage,
            "depth": int(row["depth"]),
            "target_age": row["target_age"],
            "kl_loss": float(loss.item()),
            "teacher_answer_ce": teacher_ce,
            "student_answer_ce": student_ce,
            "perceiver_grad_norm": grad_norm,
            "step_time_s": time.perf_counter() - started,
            "gpu_memory_gb": (
                torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0.0
            ),
        }
        logs.append(entry)
        if log_every and (step + 1) % log_every == 0:
            print(json.dumps(entry), flush=True)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(
        _checkpoint_payload(
            model,
            optimizer,
            stage=stage,
            completed_steps=steps,
            logs=logs,
        ),
        temporary,
    )
    os.replace(temporary, output)
    print(f"wrote {stage} checkpoint to {output}", flush=True)
    return {
        "checkpoint": str(output),
        "stage": stage,
        "steps": steps,
        "first_loss": logs[0]["kl_loss"],
        "final_loss": logs[-1]["kl_loss"],
        "logs": logs,
    }


def run_two_stage_training(
    *,
    model_name: str,
    cfg: STILLConfig,
    single_steps: int,
    recurrent_steps: int,
    learning_rate: float,
    single_output: str,
    recurrent_output: str,
    chunk_tokens: int = 64,
    seed: int = RecurrentSyntheticConfig().train_seed,
    dtype: torch.dtype = torch.bfloat16,
    log_every: int = 1,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = STILLModel(model_name, cfg=cfg, device=cfg.device, dtype=dtype)
    single_rows = generate_recurrent_rows(
        tokenizer,
        count=single_steps,
        depths=(1,),
        seed=seed,
        chunk_tokens=chunk_tokens,
    )
    single = run_training_stage(
        model,
        single_rows,
        stage="single_step",
        steps=single_steps,
        learning_rate=learning_rate,
        output_path=single_output,
        log_every=log_every,
    )

    recurrent_rows = generate_recurrent_rows(
        tokenizer,
        count=recurrent_steps,
        depths=RECURRENT_DEPTHS,
        depth_weights=RECURRENT_WEIGHTS,
        seed=seed + 1,
        chunk_tokens=chunk_tokens,
    )
    recurrent = run_training_stage(
        model,
        recurrent_rows,
        stage="recurrence_aware",
        steps=recurrent_steps,
        learning_rate=learning_rate,
        output_path=recurrent_output,
        resume_path=single_output,
        log_every=log_every,
    )
    return {"single_step": single, "recurrence_aware": recurrent}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--single-steps", type=int, default=75)
    parser.add_argument("--recurrent-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--num-latents", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--chunk-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=RecurrentSyntheticConfig().train_seed)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--single-output",
        default="/persist/cartridges/checkpoints/qwen3_4b_single_step.pt",
    )
    parser.add_argument(
        "--recurrent-output",
        default="/persist/cartridges/checkpoints/qwen3_4b_recurrent.pt",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = STILLConfig(
        model_name=args.model,
        num_latents=args.num_latents,
        latent_dim=args.latent_dim,
        num_blocks=args.num_blocks,
        lr=args.learning_rate,
        steps=args.single_steps + args.recurrent_steps,
        seed=args.seed,
        device=args.device,
    )
    result = run_two_stage_training(
        model_name=args.model,
        cfg=cfg,
        single_steps=args.single_steps,
        recurrent_steps=args.recurrent_steps,
        learning_rate=args.learning_rate,
        single_output=args.single_output,
        recurrent_output=args.recurrent_output,
        chunk_tokens=args.chunk_tokens,
        seed=args.seed,
        dtype=getattr(torch, args.dtype),
        log_every=args.log_every,
    )
    print(
        json.dumps(
            {
                stage: {key: value for key, value in details.items() if key != "logs"}
                for stage, details in result.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
