# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Training and inference codebase for **Huginn**, a depth-recurrent language model trained at scale (4096 AMD GPUs on Frontier supercomputer). Implements both standard GPT and a novel `RecurrentGPT` architecture that scales test-time compute via repeated depth iterations. Paper: https://arxiv.org/abs/2502.05171

## Key Commands

```bash
# Install
pip install -e .               # core package
pip install -e ".[dev]"         # with test dependencies
pip install -e ".[data]"        # with data preprocessing dependencies

# Training
python train.py --config=launch_configs/recurrent.yaml   # local training run
python launch_frontier.py                                  # Frontier cluster launch

# Fine-tuning
python finetuning_simple_example.py

# Evaluation (lm-eval benchmarks)
python evaluate_raven/local_lm_eval.py
python evaluate_raven/quick_checkpoint_eval.py

# Lint & type check
ruff check recpre/ train.py
pyright recpre/ train.py

# Tests
pytest                          # all tests
pytest path/to/test_file.py     # single test file
pytest -k "test_name"           # single test by name

# vLLM inference (install plugin first)
pip install -e vllm/
```

## Architecture

### Core Model (`recpre/`)

- **`model_dynamic.py`** — Two model classes: `GPT` (standard transformer) and `RecurrentGPT` (depth-recurrent variant). The recurrent model reuses a single block of transformer layers across multiple depth iterations, enabling variable compute at inference time.
- **`config_dynamic.py`** — Dataclass-based configuration: `GPTConfig` and `RecurrentConfig` with RoPE settings. These define model shape (layers, heads, embedding dim, etc.) and recurrence parameters.
- **`model_registry.py`** — 25+ pre-defined model shapes (e.g., `magpie-150m`, `nebel-raven-3.5b`). Training configs reference these by name.
- **`utils.py`** — `SimpleFabric` class: custom DDP-based distributed training without Lightning overhead. Implements custom `_allreduce_chunk_stream` to work around RCCL issues at scale. Also handles checkpoint save/load.
- **`optim.py`** — Custom optimizer `ELLISAdam` (Adam variant with atan updates, running init, tensor-wise gradient normalization, decoupled weight decay).
- **`ops.py`** — `LinearCrossEntropyLoss`: fused linear projection + cross-entropy for memory efficiency.

### Attention Backends (`recpre/attention_backends/`)

Pluggable attention system with 8+ implementations. Selected via config. Key ones:
- `amd.py` — Primary backend for Frontier (AMD GPUs)
- `cuda_flash_attention.py` — FlashAttention for NVIDIA GPUs
- `pytorch.py` — Standard SDPA fallback

### Training (`train.py`)

Single-file training orchestration (~1200 lines). Handles: fabric setup, model instantiation, data loading, training loop with gradient accumulation, validation, checkpointing, and WandB logging. Config is loaded from YAML files in `launch_configs/`.

### Configuration Flow

YAML config (`launch_configs/`) → `CLISettings` dataclass (`recpre/settings.py`) → model shape lookup from `model_registry.py` → `GPTConfig`/`RecurrentConfig` instantiation.

### Data Pipeline (`scripts/`)

Multi-step pipeline: tokenizer generation → dataset download → tokenization to parquet → shuffling. Data source definitions in `scripts/sources.yaml`. Training loads data via `recpre/huggingface_dataset.py` or parquet streams in `recpre/data_loading_utils.py`.

### Inference

- **HuggingFace compatible**: `recpre/raven_modeling_minimal.py` (standalone `PreTrainedModel` implementation)
- **vLLM plugin**: `vllm/raven_vllm.py` for fast serving
- **Checkpoint conversion**: `evaluate_raven/checkpoint_to_hf_converter.py`

## Code Style

- Python 3.11+, line length 120
- Ruff for linting (`ruff check`), Pyright for type checking (basic mode)
- Type annotations present but not exhaustive; pyright configured only for `recpre/` and `train.py`
