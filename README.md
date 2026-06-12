# Retrospective Fraud Precursor Analysis With A Stateless LSTM

This project uses the IBM Credit Card Fraud Detection dataset to classify
transaction histories as **fraud-preceding** or **normal-preceding**.

It is not ordinary next-transaction prediction. For a labeled transaction
`Tn`, the model receives only the previous `N` transactions:

```text
input  = Tn-N, ..., Tn-2, Tn-1
target = fraud label of Tn
```

`Tn` is never part of the input. The resulting probability means that the
previous sequence resembles histories observed before fraud; it does not mean
the model generated or directly predicted the next transaction.

## Leakage Controls

- Users are split 70/15/15 across train, validation, and test by default.
- Raw user IDs are used only for grouping, window creation, and splitting.
- Raw card IDs are excluded by default.
- Rolling, novelty, and velocity features use past information only.
- Encoders and scalers are fitted only on training users.
- The PyTorch LSTM receives no hidden state from earlier batches or windows.

## Setup

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the IBM transaction CSV at:

```text
data/raw/credit_card_transactions-ibm_v2.csv
```

Alternatively, update `dataset_path` in `config.yaml`.

## Run

```bash
python main.py --mode eda
python main.py --mode train
python main.py --mode evaluate
python main.py --mode predict --input path/to/sample_sequence.csv
```

The prediction CSV must contain exactly `sequence_length` chronologically
ordered transaction rows. It does not need `User` or `Is Fraud?`; prediction
mode supplies non-model placeholders for them.

## Outputs

- `outputs/models/best_model.pt`: best validation-loss checkpoint
- `outputs/models/preprocessor.joblib`: train-fitted preprocessing artifact
- `outputs/plots/`: EDA, loss, ROC, PR, and confusion-matrix plots
- `outputs/reports/metrics.json`: train/validation/test metrics
- `outputs/reports/split_users.json`: reproducible split membership
- `outputs/reports/final_report.md`: generated research report
- `outputs/logs/`: run logs

Fraud is highly imbalanced, so precision, recall, F1, PR-AUC, and MCC should be
given more weight than accuracy.

## Project Layout

```text
fraud_precursor_lstm/
  data/raw/
  data/processed/
  notebooks/01_eda.ipynb
  src/
  outputs/{models,plots,reports,logs}/
  tests/
  config.yaml
  main.py
```

## Tests

```bash
pytest
```

The focused tests cover the core research invariant: the target transaction is
excluded from every input window, and users cannot overlap across data splits.

For a quick end-to-end run without the full dataset:

```bash
python scripts/generate_synthetic_data.py
python main.py --mode train --config config.smoke.yaml
```

To build and train a compact smoke subset from the real IBM CSV:

```bash
python scripts/create_real_smoke_subset.py
python main.py --mode train --config config.real-smoke.yaml
```
