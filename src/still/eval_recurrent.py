"""Evaluate fixed-budget recurrent compaction against full and text baselines."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from still.config import STILLConfig
from still.data.recurrent_synthetic import (
    RecurrentSyntheticConfig,
    generate_evaluation_rows,
)
from still.model.wrapper import STILLModel
from still.train import kl_teacher_student

METHOD_LABELS = {
    "full_context": "Full context",
    "text_window": "Text window",
    "single_step": "Single-step Still",
    "recurrent": "Recurrent Still",
}
TARGET_AGES = ("oldest", "middle", "newest")
DEFAULT_DEPTHS = (1, 2, 4, 8, 16, 32, 100)


def _timed(fn, device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = fn()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, time.perf_counter() - started


def _answer_ce(logits: torch.Tensor, answer_ids: Sequence[int]) -> float:
    targets = torch.tensor(answer_ids, dtype=torch.long, device=logits.device)
    return float(F.cross_entropy(logits.float(), targets).item())


def _letter_token_ids(tokenizer) -> list[int]:
    """Use the same natural post-``Answer:`` tokenization as the dataset."""

    result = []
    for letter in ("A", "B", "C", "D"):
        spaced = tokenizer.encode(f" {letter}", add_special_tokens=False)
        plain = tokenizer.encode(letter, add_special_tokens=False)
        tokens = spaced if len(spaced) == 1 else plain
        if not tokens:
            raise ValueError(f"tokenizer produced no token for answer choice {letter}")
        result.append(int(tokens[0]))
    return result


def _prediction(logits: torch.Tensor, choice_token_ids: Sequence[int]) -> int:
    choices = torch.tensor(choice_token_ids, dtype=torch.long, device=logits.device)
    return int(logits[0].index_select(0, choices).argmax().item())


def _load_checkpoint(model: STILLModel, path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location=model.perceiver_device, weights_only=True)
    state = payload.get("perceiver", payload)
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    expected = {
        "num_latents": model.cfg.num_latents,
        "latent_dim": model.cfg.latent_dim,
        "num_blocks": model.cfg.num_blocks,
    }
    mismatches = {
        key: (config[key], value)
        for key, value in expected.items()
        if key in config and config[key] != value
    }
    if mismatches:
        raise ValueError(f"checkpoint/config mismatch for {path}: {mismatches}")
    model.perceiver.load_state_dict(state)
    model.perceiver.eval()
    return {
        "path": path,
        "format": payload.get("format") if isinstance(payload, dict) else None,
        "stage": payload.get("stage") if isinstance(payload, dict) else None,
        "completed_steps": payload.get("completed_steps") if isinstance(payload, dict) else None,
    }


def _record(
    *,
    row: dict,
    method: str,
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    choice_token_ids: Sequence[int],
    source_tokens: int,
    represented_source_tokens: int,
    memory_positions: int,
    compaction_seconds: float | None,
) -> dict[str, Any]:
    prediction = _prediction(logits, choice_token_ids)
    teacher_ce = _answer_ce(teacher_logits, row["answer_input_ids"])
    answer_ce = _answer_ce(logits, row["answer_input_ids"])
    depth = int(row["depth"])
    return {
        "method": method,
        "depth": depth,
        "target_age": row["target_age"],
        "example_index": int(row.get("example_index", -1)),
        "gold_index": int(row["correct_index"]),
        "prediction_index": prediction,
        "correct": int(prediction == int(row["correct_index"])),
        "answer_ce": answer_ce,
        "teacher_answer_ce": teacher_ce,
        "ce_gap_to_full": answer_ce - teacher_ce,
        "kl_to_full": float(kl_teacher_student(logits, teacher_logits).item()),
        "source_tokens": source_tokens,
        "represented_source_tokens": represented_source_tokens,
        "memory_positions": memory_positions,
        "source_tokens_per_position": source_tokens / memory_positions,
        "compaction_ms_per_chunk": (
            1000.0 * compaction_seconds / depth if compaction_seconds is not None else None
        ),
    }


def _build_recurrent_cache(model: STILLModel, chunks: Sequence[Sequence[int]]):
    cache = model.compact_tokens(chunks[0])
    for chunk in chunks[1:]:
        cache = model.recompact(cache, chunk)
    return cache


def _means(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        return statistics.fmean(float(record[key]) for record in records)

    timings = [
        float(record["compaction_ms_per_chunk"])
        for record in records
        if record["compaction_ms_per_chunk"] is not None
    ]
    return {
        "n": len(records),
        "accuracy": mean("correct"),
        "mean_answer_ce": mean("answer_ce"),
        "mean_teacher_answer_ce": mean("teacher_answer_ce"),
        "mean_ce_gap_to_full": mean("ce_gap_to_full"),
        "mean_kl_to_full": mean("kl_to_full"),
        "source_tokens": int(records[0]["source_tokens"]),
        "represented_source_tokens": int(records[0]["represented_source_tokens"]),
        "memory_positions": int(records[0]["memory_positions"]),
        "source_tokens_per_position": float(records[0]["source_tokens_per_position"]),
        "mean_compaction_ms_per_chunk": statistics.fmean(timings) if timings else None,
    }


def _aggregate(records: Sequence[dict[str, Any]], depths: Sequence[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for method in METHOD_LABELS:
        by_depth: dict[str, Any] = {}
        for depth in depths:
            depth_records = [
                record
                for record in records
                if record["method"] == method and record["depth"] == depth
            ]
            groups = {"overall": _means(depth_records)}
            for age in TARGET_AGES:
                age_records = [record for record in depth_records if record["target_age"] == age]
                if age_records:
                    groups[age] = _means(age_records)
            by_depth[str(depth)] = groups
        summary[method] = {"label": METHOD_LABELS[method], "by_depth": by_depth}
    return summary


def evaluate_recurrent_comparison(
    model: STILLModel,
    tokenizer,
    rows: Sequence[dict],
    *,
    single_checkpoint: str,
    recurrent_checkpoint: str,
    memory_positions: int,
) -> dict[str, Any]:
    """Evaluate all four arms over one deterministic held-out row set."""

    if memory_positions < 1:
        raise ValueError("memory_positions must be positive")
    depths = sorted({int(row["depth"]) for row in rows})
    choice_token_ids = _letter_token_ids(tokenizer)
    records: list[dict[str, Any]] = []
    baselines: list[tuple[dict, torch.Tensor]] = []
    model.base.eval()
    model.perceiver.eval()

    with torch.inference_mode():
        for index, row in enumerate(rows):
            chunks = row["chunks_input_ids"]
            source = [token for chunk in chunks for token in chunk]
            source_tokens = len(source)
            full_logits, full_seconds = _timed(
                lambda: model.teacher_logits(source, row["query_input_ids"], row["answer_input_ids"]),
                model.device_str,
            )
            text_source = source[-memory_positions:]
            text_logits, text_seconds = _timed(
                lambda: model.teacher_logits(
                    text_source, row["query_input_ids"], row["answer_input_ids"]
                ),
                model.device_str,
            )
            full_record = _record(
                row=row,
                method="full_context",
                logits=full_logits,
                teacher_logits=full_logits,
                choice_token_ids=choice_token_ids,
                source_tokens=source_tokens,
                represented_source_tokens=source_tokens,
                memory_positions=source_tokens,
                compaction_seconds=None,
            )
            full_record["inference_ms"] = full_seconds * 1000.0
            text_record = _record(
                row=row,
                method="text_window",
                logits=text_logits,
                teacher_logits=full_logits,
                choice_token_ids=choice_token_ids,
                source_tokens=source_tokens,
                represented_source_tokens=len(text_source),
                memory_positions=memory_positions,
                compaction_seconds=None,
            )
            text_record["inference_ms"] = text_seconds * 1000.0
            records.extend((full_record, text_record))
            baselines.append((row, full_logits.cpu()))
            print(
                json.dumps(
                    {
                        "phase": "baselines",
                        "row": index,
                        "depth": row["depth"],
                        "target_age": row["target_age"],
                        "full_correct": full_record["correct"],
                        "text_correct": text_record["correct"],
                    }
                ),
                flush=True,
            )

        checkpoint_metadata = {}
        for method, checkpoint in (
            ("single_step", single_checkpoint),
            ("recurrent", recurrent_checkpoint),
        ):
            checkpoint_metadata[method] = _load_checkpoint(model, checkpoint)
            for index, (row, teacher_cpu) in enumerate(baselines):
                chunks = row["chunks_input_ids"]
                cache, compaction_seconds = _timed(
                    lambda chunks=chunks: _build_recurrent_cache(model, chunks),
                    model.device_str,
                )
                if cache.num_latents != memory_positions:
                    raise RuntimeError(
                        f"{method} produced {cache.num_latents} positions, expected {memory_positions}"
                    )
                logits = model.decode(row["query_input_ids"], row["answer_input_ids"], cache)
                teacher_logits = teacher_cpu.to(logits.device)
                record = _record(
                    row=row,
                    method=method,
                    logits=logits,
                    teacher_logits=teacher_logits,
                    choice_token_ids=choice_token_ids,
                    source_tokens=sum(len(chunk) for chunk in chunks),
                    represented_source_tokens=sum(len(chunk) for chunk in chunks),
                    memory_positions=memory_positions,
                    compaction_seconds=compaction_seconds,
                )
                records.append(record)
                print(
                    json.dumps(
                        {
                            "phase": method,
                            "row": index,
                            "depth": row["depth"],
                            "target_age": row["target_age"],
                            "correct": record["correct"],
                            "kl_to_full": round(record["kl_to_full"], 6),
                            "compaction_ms_per_chunk": round(
                                record["compaction_ms_per_chunk"], 3
                            ),
                        }
                    ),
                    flush=True,
                )

    return {
        "checkpoint_metadata": checkpoint_metadata,
        "summary": _aggregate(records, depths),
        "records": records,
    }


def _metric_series(
    summary: dict[str, Any], method: str, depths: Sequence[int], group: str, metric: str
) -> list[float]:
    return [float(summary[method]["by_depth"][str(depth)][group][metric]) for depth in depths]


def _plot_comparison(
    summary: dict[str, Any], depths: Sequence[int], output: Path, *, group: str, metric: str,
    ylabel: str, title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "full_context": "#222222",
        "text_window": "#7f7f7f",
        "single_step": "#d95f02",
        "recurrent": "#1b9e77",
    }
    markers = {"full_context": "o", "text_window": "s", "single_step": "^", "recurrent": "D"}
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for method, label in METHOD_LABELS.items():
        ax.plot(
            depths,
            _metric_series(summary, method, depths, group, metric),
            label=label,
            color=colors[method],
            marker=markers[method],
            linewidth=2,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(depths)
    ax.set_xticklabels([str(depth) for depth in depths])
    ax.set_xlabel("Compaction depth")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if metric == "accuracy":
        ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _comparison_table(summary: dict[str, Any], depths: Sequence[int]) -> str:
    selected = [depth for depth in (1, 8, 32, 100) if depth in depths]
    blocks = []
    for group, title in (("overall", "Overall MCQ accuracy"), ("oldest", "Oldest-fact MCQ accuracy")):
        header = "| Method | KV/text positions | " + " | ".join(
            f"Depth {depth}" for depth in selected
        ) + " |"
        separator = "|---|---:|" + "---:|" * len(selected)
        lines = [f"## {title}", "", header, separator]
        for method, label in METHOD_LABELS.items():
            positions = "grows" if method == "full_context" else "64"
            values = [
                summary[method]["by_depth"][str(depth)][group]["accuracy"]
                for depth in selected
            ]
            lines.append(
                f"| {label} | {positions} | "
                + " | ".join(f"{value:.3f}" for value in values)
                + " |"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def _write_csv(summary: dict[str, Any], depths: Sequence[int], output: Path) -> None:
    fields = [
        "method", "method_label", "depth", "target_age", "n", "accuracy",
        "mean_answer_ce", "mean_teacher_answer_ce", "mean_ce_gap_to_full",
        "mean_kl_to_full", "source_tokens", "represented_source_tokens",
        "memory_positions", "source_tokens_per_position", "mean_compaction_ms_per_chunk",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in METHOD_LABELS:
            for depth in depths:
                for group, metrics in summary[method]["by_depth"][str(depth)].items():
                    writer.writerow(
                        {
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "depth": depth,
                            "target_age": group,
                            **metrics,
                        }
                    )


def write_comparison_artifacts(
    result: dict[str, Any], *, output_root: str, run_name: str, depths: Sequence[int]
) -> dict[str, str]:
    """Write a new run atomically; never replace an existing result or plot directory."""

    root = Path(output_root)
    metrics_root = root / "metrics"
    plots_root = root / "plots"
    metrics_target = metrics_root / run_name
    plots_target = plots_root / run_name
    if metrics_target.exists() or plots_target.exists():
        raise FileExistsError(f"refusing to overwrite existing Phase 3 run: {run_name}")
    metrics_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)
    metrics_tmp = Path(tempfile.mkdtemp(prefix=f".{run_name}.", dir=metrics_root))
    plots_tmp = Path(tempfile.mkdtemp(prefix=f".{run_name}.", dir=plots_root))

    (metrics_tmp / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (metrics_tmp / "comparison_table.md").write_text(
        _comparison_table(result["summary"], depths)
    )
    _write_csv(result["summary"], depths, metrics_tmp / "comparison.csv")
    _plot_comparison(
        result["summary"], depths, plots_tmp / "oldest_accuracy_vs_depth.png",
        group="oldest", metric="accuracy", ylabel="Oldest-fact MCQ accuracy",
        title="Fixed-budget recurrent memory: oldest-fact retention",
    )
    _plot_comparison(
        result["summary"], depths, plots_tmp / "overall_accuracy_vs_depth.png",
        group="overall", metric="accuracy", ylabel="Overall MCQ accuracy",
        title="Fixed-budget recurrent memory: overall accuracy",
    )
    _plot_comparison(
        result["summary"], depths, plots_tmp / "overall_ce_gap_vs_depth.png",
        group="overall", metric="mean_ce_gap_to_full", ylabel="CE gap to full-context teacher",
        title="Answer CE degradation relative to full context",
    )
    os.replace(metrics_tmp, metrics_target)
    os.replace(plots_tmp, plots_target)
    return {"metrics": str(metrics_target), "plots": str(plots_target)}


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--depths", type=int, nargs="+", default=list(DEFAULT_DEPTHS))
    parser.add_argument("--per-depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=RecurrentSyntheticConfig().eval_seed)
    parser.add_argument("--chunk-tokens", type=int, default=64)
    parser.add_argument("--memory-positions", type=int, default=64)
    parser.add_argument("--num-latents", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument(
        "--single-checkpoint",
        default="/persist/cartridges/checkpoints/qwen3_4b_single_step.pt",
    )
    parser.add_argument(
        "--recurrent-checkpoint",
        default="/persist/cartridges/checkpoints/qwen3_4b_recurrent.pt",
    )
    parser.add_argument("--output-root", default="/persist/cartridges")
    parser.add_argument("--run-name", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.per_depth < 1:
        raise ValueError("per-depth must be positive")
    if len(set(args.depths)) != len(args.depths) or any(depth < 1 for depth in args.depths):
        raise ValueError("depths must be unique positive integers")
    if args.memory_positions != args.num_latents:
        raise ValueError("memory-positions and num-latents must match for the equal-budget comparison")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cfg = STILLConfig(
        model_name=args.model,
        num_latents=args.num_latents,
        latent_dim=args.latent_dim,
        num_blocks=args.num_blocks,
        device=args.device,
    )
    model = STILLModel(
        args.model,
        cfg=cfg,
        device=args.device,
        dtype=_dtype(args.dtype),
        attn_implementation="sdpa",
    )
    rows = generate_evaluation_rows(
        tokenizer,
        per_depth=args.per_depth,
        seed=args.seed,
        depths=args.depths,
        chunk_tokens=args.chunk_tokens,
    )
    for index, row in enumerate(rows):
        row["example_index"] = index
    comparison = evaluate_recurrent_comparison(
        model,
        tokenizer,
        rows,
        single_checkpoint=args.single_checkpoint,
        recurrent_checkpoint=args.recurrent_checkpoint,
        memory_positions=args.memory_positions,
    )
    result = {
        "format": "still-recurrent-eval-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "depths": args.depths,
            "per_depth": args.per_depth,
            "seed": args.seed,
            "chunk_tokens": args.chunk_tokens,
            "memory_positions": args.memory_positions,
            "num_latents": args.num_latents,
            "latent_dim": args.latent_dim,
            "num_blocks": args.num_blocks,
        },
        **comparison,
    }
    artifacts = write_comparison_artifacts(
        result,
        output_root=args.output_root,
        run_name=args.run_name,
        depths=args.depths,
    )
    print(json.dumps({"artifacts": artifacts, "config": result["config"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
