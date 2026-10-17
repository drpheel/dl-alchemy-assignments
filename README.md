# CS312 Deep Learning Alchemy (local setup)

This checkout trains small autoregressive language models on a prepared tokenized
DCLM dataset using **your own Modal workspace** (not the Stanford course
workspace). You pay for Modal GPU time and storage yourself.

For model / data / optimizer details, see [MACHINE_LEARNING.md](MACHINE_LEARNING.md).  
For local CUDA / Slurm without Modal, see [gpu/README.md](gpu/README.md).  
Upstream course docs also cover own-workspace Modal in
[README_MODAL_OWN_WORKSPACE.md](README_MODAL_OWN_WORKSPACE.md); **this README is
the customized path for this machine.**

---

## Local machine notes (Intel Mac)

This Mac is `x86_64` Darwin. The repo pins `torch==2.6.0`, which has **no Intel
Mac wheels**, so the local venv uses:

- `torch==2.2.2` (Intel Mac–compatible)
- `numpy<2` (compatible with that torch)

Modal jobs still build and run on Linux GPUs with the course lockfile. Local
torch is only for launching / importing.

**Always use `--no-sync` with `uv run`**, or uv will try to reinstall torch
2.6.0 and fail:

```bash
uv run --no-sync <command>
```

Optional shell alias:

```bash
alias uvrun='uv run --no-sync'
```

If you recreate the venv from scratch on this Mac:

```bash
uv python install 3.10
uv sync --no-install-package torch   # needs Rust on PATH for cbor2 (modal dep)
uv pip install "torch==2.2.2" "numpy<2"
```

---

## 1. Prerequisites

- terminal, `git`, `uv`
- your own [Modal](https://modal.com/) account with GPU access
- a [W&B](https://wandb.ai/) account if you want online logging
- ~25 GB free disk once, to download and upload the public dataset

## 2. Configure `utils.py`

Edit the student-facing block near the top of [utils.py](utils.py). Use **your**
Modal environment (often `main`, or one you create). Add the fourth line so data
is read from your environment instead of `cs312-shared-data`:

```python
CONFIG_MODAL_ENVIRONMENT = "my-environment"
CONFIG_WANDB_ENTITY = "YOUR_WANDB_USERNAME_OR_TEAM"
CONFIG_WANDB_PROJECT = "assignments"
MODAL_SHARED_DATA_ENVIRONMENT = CONFIG_MODAL_ENVIRONMENT
```

Replace `my-environment` and the W&B entity with yours. Leave
`CONFIG_WANDB_PROJECT` as `assignments` unless you want a different project name.

## 3. Authenticate Modal

```bash
uv run --no-sync modal setup
uv run --no-sync modal environment list
```

Create an environment if you need one, then select it:

```bash
uv run --no-sync modal environment create my-environment
uv run --no-sync modal config set-environment my-environment
```

Use the same name as `CONFIG_MODAL_ENVIRONMENT`.

## 4. Configure W&B (online logging)

Create a Modal Secret in that environment (do not put the key in source files):

```bash
uv run --no-sync modal secret create --env my-environment --force dl-alchemy-wandb WANDB_API_KEY=YOUR_WANDB_API_KEY
```

Only needed for runs with `wandb_online=True`.

## 5. Download And Upload Data (once)

Allow ~25 GB free local disk. Download the public release
[`kothasuhas/dl_alchemy_seq9p6m_context1024`](https://huggingface.co/datasets/kothasuhas/dl_alchemy_seq9p6m_context1024)
(~19.7 GB; no HF account required):

```bash
uv run --no-sync python -m download_data --output-dir ./data/dclm_9p6m_ctx1024
```

Create a Volume in your environment and upload (once):

```bash
uv run --no-sync modal volume create --env my-environment --version 2 hard-dl-dclm-v1
uv run --no-sync modal volume put --env my-environment hard-dl-dclm-v1 ./data/dclm_9p6m_ctx1024 /datasets/dclm_9p6m_ctx1024
```

Check layout (expect `manifest.json`, `train/`, `val/`):

```bash
uv run --no-sync modal volume ls --env my-environment hard-dl-dclm-v1 /datasets/dclm_9p6m_ctx1024
```

After a successful upload you can delete the local `./data/...` copy. Keep the
Modal Volume for later runs. Billing is on your Modal account:
[modal.com/pricing](https://modal.com/pricing).

## 6. Smoke Test

Modal requires a **payment method** on your account before any GPU job
(H100, T4, etc.) will launch. Add one at
[modal.com/settings/billing](https://modal.com/settings/billing).

```bash
uv run --no-sync python -m experiments.smoke.modal_smoke_train
```

Default recipe: d8, 600k train sequences, ctx 1024, batch 64, AdamW lr 0.003,
linear decay, data seed 42, H100. Expect ~10 minutes of training once the job is
running; cold start / image build can take longer the first time.

Open the printed Modal app link for logs. Checkpoints land under `/root/data/ckpts`
on your writable Modal Volume.

## 7. Change Training Configs

Inspect `TrainConfig` in [train.py](train.py). Typical overrides:

```python
TrainConfig(
    learning_rate=3e-3,
    lr_schedule="linear",
    num_train_sequences=600_000,
    batch_size=64,
    model_name="d8",
    data_seed=42,
    run_name_suffix="my-experiment",
    wandb_tags=("my-experiment",),
)
```

Launchers build one or more `TrainConfig`s and pass them to
`launch_training_jobs`. Prefer Python config over new CLI flags.

## 8. LR Tuning Template

```text
experiments/lr_tuning/launch_lr_tuning.py
experiments/lr_tuning/plot_lr_tuning.py
```

```bash
uv run --no-sync python -m experiments.lr_tuning.launch_lr_tuning
uv run --no-sync python -m experiments.lr_tuning.plot_lr_tuning
```

Use a fresh `EXPERIMENT_KEY` when you change the sweep so plots do not mix old
and new W&B runs.

## 9. Useful Commands

```bash
uv run --no-sync python -m scripts.modal_usage
uv run --no-sync modal environment billing summary --for "this month"
uv run --no-sync modal secret list --env my-environment
uv run --no-sync modal dashboard
uv run --no-sync python -m scripts.inspect_data_row
```

## 10. Data Layout On Modal

- **Read-only data Volume** (`hard-dl-dclm-v1`): mounted for training input
- **Writable Volume** (`volume-dl_alchemy`): checkpoints / models under `/root/data`

This is the **base** dataset (not a pre-shuffled seed cache). With default
`data_seed=42`, training materializes a shuffled subset on temporary storage
inside the job; that does not rewrite your persistent Volume.
