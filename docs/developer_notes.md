# Developer Notes

## Dataset configuration

Dataset-specific constants live in `trajgpt_config.py`.

Use `get_dataset_spec(env)` or `DATASET_SPECS[name]` instead of hard-coding:

- action dimensions,
- spatial vocabulary sizes,
- user vocabulary sizes,
- departure-time bins,
- speed bins.

This keeps the implementation aligned with the paper appendix and makes dataset-specific assumptions easy to audit.

## Model outputs

Transformer `forward()` methods return action logits for training. `get_action()` applies softmax for generation/evaluation. Do not apply softmax before `nn.CrossEntropyLoss`.

## Environment semantics

`Env/*` classes use `done=True` for invalid moves, terminal actions, and unreachable transitions. This avoids infinite generation loops and keeps evaluation behavior explicit.
