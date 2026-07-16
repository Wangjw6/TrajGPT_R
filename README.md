# TrajGPT-R

This repository contains the implementation for:

> Jiawei Wang, Chuang Yang, Jiawei Yong, Xiaohang Xu, Hongjun Wang, Noboru Koshizuka, Shintaro Fukushima, Ryosuke Shibasaki, Renhe Jiang. "TrajGPT-R: Generating urban mobility trajectory with reinforcement Learning-Enhanced generative Pre-trained transformer." Transportation Research Part C, 192, 105862, 2026.

TrajGPT-R models urban trajectory generation as an offline reinforcement-learning sequence modeling problem. The code includes:

- GPT-style trajectory pretraining with state, action, and arrival-status tokens.
- Inverse-reinforcement-learning reward models for trajectory preference estimation.
- Reward-model fine-tuning for generated trajectory reliability and diversity.
- Evaluation utilities for path similarity, BLEU, link/connection JSD, and diversity metrics.

## Paper Alignment

The main entry points map to the paper datasets as follows:

| Dataset | Entrypoint | Default model | Key paper settings |
| --- | --- | --- | --- |
| Toyota | `main.py` | `my` | embedding dim 512, context length 64, action dim 9 |
| T-Drive | `main_drive.py` | `my` | embedding dim 256, context length 12, action dim 10 |
| Porto | `main_porto.py` | `my` | embedding dim 256, context length 64, action dim 10 |

`model_type=my` trains/evaluates the trajectory generator. `model_type=myp` enables reward-model fine-tuning and requires compatible trajectory and reward-model checkpoints when `--phase` selects a fine-tuning phase.

Dataset-specific constants are centralized in `trajgpt_config.py`; see `docs/developer_notes.md` before adding a new dataset or changing vocabulary/action dimensions.

## Installation

The original experiments used PyTorch and Transformers versions from the Decision Transformer-era stack:

```bash
pip install -r requirements.txt
```

CUDA is optional for import and basic CPU checks, but full training is expected to require a CUDA-capable GPU.

## Data And Checkpoints

Do not assume that all datasets and checkpoints in a local research workspace are redistributable. Before publishing:

- Release only datasets whose licenses permit redistribution.
- For private Toyota data, publish only derived instructions or anonymized/preprocessed artifacts that are explicitly approved for release.
- Release large model checkpoints through GitHub Releases, Zenodo, Hugging Face, or another artifact host, with matching license and provenance notes.
- Keep generated outputs under `results/`, which is ignored by Git.

## Reproducibility

Example commands:

```bash
python main.py --model_type my --env toyota --K 64 --embed_dim 512 --batch_size 64 --n_layer 2 --weight_decay 0.05
python main_drive.py --model_type my --env tdrive --K 12 --embed_dim 256 --batch_size 64 --n_layer 2 --weight_decay 0.02
python main_porto.py --model_type my --env porto --K 64 --embed_dim 256 --batch_size 128 --n_layer 3 --weight_decay 0.05
```

Checkpoint loading is explicit:

```bash
python main.py --model_type my --phase 1 --pretrained_checkpoint saved_models/toyota_generator.pt
python main.py --model_type myp --phase 1 --pretrained_checkpoint toyota_generator --preference_checkpoint toyota_iql_reward
```

`--pretrained_checkpoint` accepts either a path or a stem under `saved_models/` with `.pt` appended. `--preference_checkpoint` accepts either a path or a stem under `save_preference/` with `.pth` appended. The code no longer falls back to unpublished local checkpoint names.

For academic use, cite the paper and this repository. If you use third-party datasets, cite their original sources as well.

## Research Integrity Notes

This code includes third-party-derived components, especially the GPT-2 implementation adapted from Hugging Face Transformers under the Apache-2.0 license. See `THIRD_PARTY_NOTICES.md`.

Before public release, confirm the code license with all copyright holders and collaborators. A repository is not fully open source until it has an explicit OSI-compatible license.
