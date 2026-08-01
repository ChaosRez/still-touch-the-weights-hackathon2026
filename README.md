# STILL: Neural KV Cache Compaction (HuggingFace-native)

A standalone reimplementation of STILL (arXiv:2606.07878) as offline forward-KL
distillation against a frozen Qwen3 base model, in pure HuggingFace `transformers`.

A per-layer Perceiver compresses each layer's full `[K;V]` into a fixed-size compact
cache (`t` latents, `t << T`) plus a per-position attention bias. The frozen base model
decodes against the compact cache as if it were real context. Only the perceiver is
trained; the base model is frozen.

## Status: basic loop + recurrent training (v1)

This package ships the **basic loop**: it validates the full plumbing (perceiver
compresses per-layer K/V, the frozen model decodes against the compact cache, KL
gradients flow to the perceiver only, eval runs) **without** the paper's three
correctness fixes (RoPE strip/re-apply, final-RMSNorm removal, identity init). v1
answers "is the machinery wired and do gradients flow," not "does STILL hit the paper's
MCQ numbers." The three fixes are a documented follow-up.

The recurrent extension keeps exactly 64 synthetic KV positions across repeated
compactions. It provides deterministic access-code MCQs and a two-stage trainer: depth-1
distillation followed by last-step-gradient training on a stratified 10/20/30/40 mix of
recurrence depths 1/2/4/8.

## Quickstart

```bash
uv sync
uv run pytest -q                       # CPU unit tests (tiny random-weight model)

# Build an on-disk cache of on-policy QuALITY rows (tiny model for a fast smoke):
uv run still-preprocess --split val --limit 16 --out /tmp/still-cache \
  --model hf-internal-testing/tiny-random-Qwen3ForCausalLM

# Train the perceiver (5-step KL loop):
uv run still-train --dataset /tmp/still-cache --steps 20 \
  --model hf-internal-testing/tiny-random-Qwen3ForCausalLM --ckpt-dir /tmp/still-ckpt

# Evaluate (MCQ acc / CE utilization / compression / peak memory):
uv run still-eval --dataset /tmp/still-cache --ckpt /tmp/still-ckpt/perceiver_final.pt \
  --model hf-internal-testing/tiny-random-Qwen3ForCausalLM --out /tmp/still-metrics.json

# Fixed-budget recurrent comparison checkpoints (one H100 for Qwen3-4B):
uv run still-train-recurrent --model Qwen/Qwen3-4B \
  --single-steps 50 --recurrent-steps 75 --num-latents 64 \
  --single-output /tmp/qwen3_4b_single_step.pt \
  --recurrent-output /tmp/qwen3_4b_recurrent.pt
```

For the real 0.6B run, drop the `--model` overrides (defaults to `Qwen/Qwen3-0.6B`).

## Layout

- `src/still/config.py` — `STILLConfig` dataclass.
- `src/still/data/` — QuALITY loading, on-policy teacher answer generation, preprocess CLI.
- `src/still/model/perceiver.py` — the per-layer Perceiver.
- `src/still/model/attention.py` — registered biased attention forward + `CompactCache`.
- `src/still/model/wrapper.py` — `STILLModel` (frozen base + perceiver orchestration).
- `src/still/train.py` — the distillation loop.
- `src/still/data/recurrent_synthetic.py` — deterministic repeated-compaction facts.
- `src/still/train_recurrent.py` — single-step and recurrence-aware checkpoint training.
- `src/still/eval.py` — the eval harness.

## Deferred

The three correctness fixes (RoPE strip/re-apply with internal latent RoPE, final-RMSNorm
removal before output projections, identity init + content-stripping biases), full recurrent
evaluation, and vLLM serving integration.

## Reference

- STILL paper: arXiv:2606.07878
- Dataset: https://huggingface.co/datasets/emozilla/quality
