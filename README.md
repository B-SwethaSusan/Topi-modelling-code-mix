# XLM-RoBERTa Code-Mixed Text Classification

Telugu-English code-mixed text classification into 5 categories using XLM-RoBERTa.

## Categories
- Education
- Entertainment
- Politics
- Sports
- Miscellaneous

## Project Structure

```
major_project/
├── data/                       # Dataset files
│   ├── train_dataset.csv
│   ├── test_dataset.csv
│   └── labeled_dataset_33906.csv
├── src/                        # Source modules
│   ├── config.py               # Configuration
│   ├── utils.py                # Utilities
│   ├── preprocessing.py        # Data preprocessing
│   ├── dataset.py              # PyTorch Dataset
│   ├── model.py                # XLM-RoBERTa model
│   ├── train.py                # Training loop
│   └── evaluate.py             # Evaluation
├── outputs/                    # Results & models
│   ├── models/                 # Saved checkpoints
│   ├── logs/                   # Training logs
│   └── results/                # Evaluation reports
├── train.py                    # Main training script
├── evaluate.py                 # Standalone evaluation
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train.py
```

### 3. Evaluate
```bash
python evaluate.py --model_path outputs/models/best_model.pt
```

## Model Architecture

- **Base Model**: XLM-RoBERTa-base (278M params)
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Layers**: 12
- **Max Sequence**: 128 tokens
- **Classification Head**: Linear (768 -> 5)

## Training Configuration

- **Batch Size**: 32
- **Learning Rate**: 2e-5
- **Epochs**: 15 (with early stopping)
- **Optimizer**: AdamW (differential LR)
- **Scheduler**: Linear warmup
- **Gradient Accumulation**: 2 steps
- **Early Stopping**: Patience 3

## Data Preprocessing

- Remove URLs, emails, mentions
- Normalize repeated characters
- Keep emojis (contextual)
- WordPiece tokenization
- Handle class imbalance (optional oversampling)

## Evaluation Metrics

- Accuracy
- Precision, Recall, F1 (macro & weighted)
- Per-class breakdown
- Confusion matrix
- Training curves

## Files

| File | Description |
|------|-------------|
| `train.py` | Main training entry point |
| `evaluate.py` | Standalone evaluation |
| `src/config.py` | All hyperparameters |
| `src/model.py` | Model definition |
| `src/train.py` | Training loop & early stopping |
| `src/evaluate.py` | Metrics & visualization |

## Performance Target

- Target: **>= 78% accuracy**
- Weighted F1-score tracking
- Per-class analysis

## Reproducibility

Seed 42 set across all random generators:
- Python `random`
- NumPy
- PyTorch
- CUDA (if available)

## Citation

XLM-RoBERTa: https://arxiv.org/abs/1911.02116
