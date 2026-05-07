# Banking77 Intent Classification — PEFT Finetuning Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Unsloth](https://img.shields.io/badge/Unsloth-Optimized-brightgreen?logo=lightning&logoColor=white)](https://github.com/unslothai/unsloth)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%2FDoRA-blueviolet)](https://github.com/huggingface/peft)
[![Weights & Biases](https://img.shields.io/badge/Weights%20%26%20Biases-Tracking-lightblue?logo=weightsandbiases&logoColor=white)](https://wandb.ai/)
[![Kaggle](https://img.shields.io/badge/Kaggle-P100%2FT4-20BEFF?logo=kaggle&logoColor=white)](https://kaggle.com/)

MLOps PEFT fine-tuning pipeline for Banking77 intent classification using **Llama-3.2-1B-Instruct**, optimized with **Unsloth**.

## 1. Overview

This project implements parameter-efficient fine-tuning (PEFT) of Llama-3.2-1B-Instruct for classifying banking customer queries into 77 intent categories. The pipeline leverages **DoRA** (Decomposed Rank Adaptation) and **rsLoRA** (Rank-Stabilized LoRA) to improve training stability, parameter efficiency, and low-memory adaptation.

Training was conducted on **Kaggle P100/T4 GPUs** for 3 epochs using 9K deduplicated samples.

---

## 2. Dataset

**Banking77** is a benchmark intent detection dataset for banking domain:
- **Total Examples:** 13,077 (10,003 train + 3,080 test after dedup)
- **Intent Classes:** 77 distinct banking intents (e.g., "activate_my_card", "cancel_transfer", "lost_or_stolen_card")
- **Format:** Text queries with single-label intent classification
- **Source:** https://huggingface.co/datasets/banking77

After preprocessing (deduplication + 90/10 split):
- **Training:** 9,000 examples
- **Validation:** 1,000 examples
- **Test:** 3,077 examples


### Data Processing
- **Deduplication:** Removes 1K duplicate examples for cleaner training
- **Train/Val Split:** 90/10 stratified split (9K train, 1K val)
- **Chat Template Formatting:** Llama-3-specific format with system/user/assistant roles
- **Prompt Structure:**
  ```
  System: "You are a banking intent classifier..."
  User: "Classify: <customer_query>"
  Assistant: "<intent_label>"
  ```

---

## 3. Key Features

### Model Architecture
- **Base Model:** `unsloth/Llama-3.2-1B-Instruct` (1B parameters, chat-optimized)
- **Quantization:** 4-bit NF4 (QLoRA) — reduces VRAM from 8GB to ~2GB
- **Max Sequence Length:** 512 tokens

### PEFT Configuration
- **LoRA Rank:** 32 (higher capacity for domain-specific knowledge)
- **DoRA:** Weight-decomposed rank adaptation for stable, smooth training dynamics
- **rsLoRA:** Rank-stabilized scaling with adaptive normalization (better gradient flow)
- **LoRA Dropout:** 0.05 (regularization to prevent overfitting on small dataset)
- **Target Modules:** All attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP projections (`gate_proj`, `up_proj`, `down_proj`)

### Training Techniques
- **Response-Only Loss Masking:** Model learns exclusively on assistant responses, not on prompts
- **Sequence Packing:** Multiple short sequences batched into single forward pass for efficiency
- **Gradient Accumulation:** Effective batch size 32 (8 per-device × 4 steps) for stable updates
- **Cosine LR Scheduling:** Smooth decay with 5% warmup
- **8-bit AdamW:** Memory-efficient optimizer
- **Gradient Checkpointing:** Unsloth's optimized implementation for minimal VRAM overhead

### Infrastructure & Tracking
- **W&B Integration:** Full experiment tracking, hyperparameter logging, training curves
- **Checkpoint Management:** Keeps 3 best checkpoints, saves every epoch
- **Structured Logging:** File-based logging for debugging and audit trails

---

## 4. Evaluation & Results

### 4.1. Comparison Baseline

This project is compared against the following public implementation:
- **Repository:** [Fine-Tuning-Llama-3.2-1B-Instruct-on-Banking77-Intent-Classification](https://github.com/rajo69/Fine-Tuning-Llama-3.2-1B-Instruct-on-Banking77-Intent-Classification)
- **Hugging Face Model:** [rajo0113/banking77-llama-1b-lora](https://huggingface.co/rajo0113/banking77-llama-1b-lora)

### 4.2. Performance Metrics Comparison

| Metric | Advanced PEFT (DoRA+rsLoRA) | LoRA Baseline | Improvement |
|--------|-----------|-----------|-------------|
| **Exact Match Accuracy** | **92.31%** | 87.44% | **+4.87%** ✓ |
| **Macro F1 Score** | **0.8377** | 0.7493 | **+0.0884** ✓ |
| **Macro Precision** | **0.8420** | 0.7567 | **+0.0853** ✓ |
| **Macro Recall** | **0.8362** | 0.7481 | **+0.0881** ✓ |

**Key Findings:**
- **Advanced PEFT outperforms LoRA Baseline** on all metrics
- **+4.87% accuracy improvement** demonstrates weight decomposition effectiveness
- **F1 score improvement** (+0.0884) indicates DoRA better balances precision and recall
- **Consistent improvements across all metrics** validate DoRA + rsLoRA + NEFTune synergy

### 4.3. Error Analysis

#### Top Confusion Patterns (Advanced PEFT)

The Advanced PEFT model's **92.31% accuracy** shows minimal confusion. Most errors occur between semantically similar intents:

| True Intent | Predicted As | Count | Reason |
|------------|-------------|-------|--------|
| `fiat_currency_support` | `exchange_via_app` | 7 | Similar currency/exchange context |
| `card_arrival` | `card_delivery_estimate` | 6 | Overlapping delivery-related language |
| `top_up_by_bank_transfer_charge` | `transfer_fee_charged` | 6 | Fee-related terminology confusion |
| `balance_not_updated_after_bank_transfer` | `pending_transfer` | 6 | Both relate to transfer delays |
| `transfer_not_received_by_recipient` | `transfer_not_received` | 5 | Near-duplicate intent phrasing |

**Interpretation:** Confusions primarily occur between **intent pairs with overlapping semantic fields**, indicating the model has learned most distinctions correctly.

#### Top Confusion Patterns (LoRA Baseline)

LoRA Baseline shows higher confusion, with **87.44% accuracy**:

| True Intent | Predicted As | Count | Reason |
|------------|-------------|-------|--------|
| `Refund_not_showing_up` | `refund_not_showing_up` | 39 | **Case sensitivity / label inconsistency** ⚠️ |
| `why_verify_identity` | `why_is_my_card_blocked` | 37 | **Semantic similarity** (security checks) |
| `fiat_currency_support` | `exchange_via_app` | 8 | Currency/exchange context |
| `topping_up_by_card` | `top_up_reverted` | 8 | Top-up operation confusion |
| `beneficiary_not_allowed` | `failed_transfer` | 8 | Transfer failure reason ambiguity |

**Key Observations:**
- LoRA Baseline struggles more with semantically adjacent intents
- Large confusion counts (39, 37) suggest potential label preprocessing issues
- LoRA Baseline's lower recall on minority classes impacts overall performance

### 4.4. Advanced PEFT Pipeline: DoRA + rsLoRA + NEFTune

**Why Advanced PEFT performs better:**
1. **Weight Decomposition (DoRA):** Separates magnitude from direction, allowing more stable gradient flow
2. **Rank Stabilization (rsLoRA):** Adaptive normalization ensures consistent learning across 77 intent classes
3. **Noise Injection (NEFTune):** α=5 noise injection improves instruction-following robustness on small datasets
4. **Improved Regularization:** LoRA dropout (0.05) + DoRA decomposition prevents overfitting
5. **Training Stability:** Smoother loss curves and fewer training instabilities observed

**Comparison Summary:**
- ✅ **Advanced PEFT:** Better for precision-critical applications (financial domain) — 92.31% accuracy
- ⚠️ **LoRA Baseline:** Acceptable but higher error rate on minority classes — 87.44% accuracy

---

## 5. Project Structure

```
├── configs/                          # YAML configuration files
│   ├── train_config.yaml             # Training hyperparameters & PEFT settings
│
├── scripts/                          # Main training pipeline
│   ├── data_process/                 # Data preparation
│   │   ├── load_data.py              # Download & load Banking77 from HuggingFace
│   │   ├── process_data.py           # Deduplication, train/val/test splits
│   │   └── format_data.py            # Apply Llama chat template formatting
│   │
│   ├── finetune/                     # Model fine-tuning
│   │   ├── merge.py                  # Merge DoRA adapters into base model
│   │   └── notebooks/
│   │       ├── finetune-llama-3-2-1b-instruct-on-banking77.ipynb  # Kaggle notebook
│   │       └── merge.ipynb           # Model merging notebook
│   │
│   ├── eval/                    
│   │   └── notebooks/
│   │       ├── benchmark-intent-classifiaction-dora-vs-lora.ipynb # Metrics comparison notebook
│   │       └── evaluation-latent-space-visualization.ipynb           # t-SNE/UMAP visualization notebook
│   │
│   ├── utils/                        # Shared utilities
│   │   ├── constants.py              # Banking77 label mappings, prompts
│   │   ├── io_utils.py               # File I/O helpers
│   │   ├── logging_config.py         # Structured logging setup
│   │   └── seed.py                   # Reproducibility utilities
│
├── data/                             # Data artifacts
│   ├── raw/                          # Original Banking77 dataset
│   │   ├── train/, test/             # Arrow format datasets
│   │   └── dataset_dict.json         # DatasetDict manifest
│   ├── processed/
│   │   ├── split/                    # Train/val/test splits (deduplicated)
│   │   └── template/                 # Chat-formatted prompt data
│
├── models/                           # Model artifacts
│   ├── base/                         # Base model cache
│   ├── checkpoints/                  # Training checkpoints (best 3 kept)
│   └── finetuned/                    # Final merged model (base + DoRA adapters)
│
├── outputs/                          # Training outputs & analysis
│   ├── logs/                         # Training & data processing logs
│   ├── plots/			              # Training curves, loss plots
│   ├── error_analysis/               # Misclassified examples, confusion matrices
│   └── metrics.json                  # Evaluation metrics
│
├── requirements.yml                  # Conda environment specification
└── README.md                         # This file
```

---

## 6. Quick Start

### 6.1. Setup Environment

```bash
# Clone repository
git clone <repo_url>
cd Finetuning-Intent-Detection-Model

# Create Conda environment
conda env create -f requirements.yml
conda activate banking77-finetuning
```

### 6.2. Data Preparation

To regenerate datasets from scratch:

```bash
# 1. Download Banking77 dataset from HuggingFace
python scripts.data_process.load_data.py
# 2. Deduplicate & split (90/10 train/val, keep original test)
python scripts.data_process.process_data.py
# 3. Apply Llama chat template formatting
python scripts.data_process.format_data.py
```

### 6.3. Fine-tuning

Use the Kaggle notebook for interactive fine-tuning:

**File:** [finetune-llama-3-2-1b-instruct-on-banking77.ipynb](scripts/finetune/notebooks/finetune-llama-3-2-1b-instruct-on-banking77.ipynb)


**Configuration is in `configs/train_config.yaml`:**

---

## 7. References & Citations

### Dataset

**Banking77: A New Dataset for State-of-the-art Intent Detection**
   - Paper: [Banking77-paper](https://arxiv.org/abs/2003.04807)
   - Dataset: [Banking77-dataset](https://huggingface.co/datasets/PolyAI/banking77)

### Papers

1. **DoRA: Weight-Decomposed Low-Rank Adaptation**
   - Arxiv: [DoRA-paper](https://arxiv.org/abs/2402.09353)
   - GitHub: [DoRA-github](https://github.com/NVlabs/DoRA)
   - Key contribution: Weight decomposition into magnitude and direction for stable training

2. **QLoRA: Efficient Finetuning of Quantized LLMs**
   - Paper: [QLoRA-paper](https://arxiv.org/abs/2305.14314)
   - 4-bit quantization technique used in this pipeline

### Libraries & Frameworks

- **PEFT (Parameter-Efficient Fine-Tuning):** [github.com/huggingface/peft](https://github.com/huggingface/peft)
- **Unsloth:** [unsloth-github](https://github.com/unslothai/unsloth) (GPU optimization)
- **TRL (Transformer Reinforcement Learning):** [TRL-github](https://github.com/huggingface/trl)
- **Hugging Face Transformers:** [Hugging Face-docs](https://huggingface.co/docs/transformers)
- **Weights & Biases:** [wandb.ai](https://wandb.ai) (Experiment tracking)

### Related Work & Baselines

- **Existing Repository (LoRA Baseline):**
  - **Repository:** [Fine-Tuning-Llama-3.2-1B-Instruct-on-Banking77-Intent-Classification](https://github.com/rajo69/Fine-Tuning-Llama-3.2-1B-Instruct-on-Banking77-Intent-Classification)
  - **Hugging Face Model:** [rajo0113/banking77-llama-1b-lora](https://huggingface.co/rajo0113/banking77-llama-1b-lora)
  - Standard LoRA approach for Banking77 classification
  - Inspired the comparison study in this project

---

## 🤝 Contributing

Contributions welcome! Please open issues or PRs for improvements.
