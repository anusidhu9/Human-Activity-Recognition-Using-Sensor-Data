# Human Activity Recognition Using Sensor Data

Classifying six activities (walking, walking upstairs, walking downstairs, sitting,
standing, laying) from smartphone accelerometer and gyroscope signals in the
[UCI HAR dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).

Three models are compared on the official subject-independent train/test split:

| Model | Input |
|---|---|
| Random Forest (baseline) | 561 engineered features per window |
| 1D CNN | raw inertial window, 9 channels x 128 timesteps |
| LSTM | raw inertial window, 128 timesteps x 9 channels |

Experiments, predictions, metrics and error analysis are stored in a normalized
SQLite database (SQLAlchemy).

> **Status:** work in progress, built in stages. Stages 0-6 are done (environment, dataset, SQLite database, data pipeline,
> Random Forest, 1D CNN, LSTM - one seed each). Multi-seed comparison, figures and
> the full write-up follow in Stages 7-8.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Linux/macOS)
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build
pip install -r requirements.txt
pip install -e .
pytest

python scripts/download_data.py   # downloads + extracts UCI HAR into data/raw/ (skips if present)
python scripts/inspect_data.py    # shapes, class balance, subject split checks
python scripts/build_db.py        # creates db/har.sqlite with dataset/sample metadata (refuses to overwrite)
python scripts/train.py --model rf --seed 42
python scripts/train.py --model cnn --seed 42   # add --device cuda/cpu to force a device
python scripts/tune_lstm.py --seed 42            # validation-only LSTM search -> artifacts/tuning/
python scripts/train.py --model lstm --seed 42   # uses the tuned defaults
```

## Results so far

Single run, seed 42. Trained on 5,952 windows from 17 subjects, model selected on
4 held-out training subjects (3, 16, 22, 23), evaluated once on the official test set
(2,947 windows, 9 unseen subjects). CPU, 4 threads.

| Model | Test accuracy | Macro F1 | Train time | Inference |
|---|---|---|---|---|
| Random Forest (300 trees, max_features=sqrt, min_samples_leaf=5) | 0.9097 | 0.9073 | 14.2 s | 0.063 ms/window |
| 1D CNN (3 conv blocks, 127k params, best epoch 23 of 33) | 0.9250 | 0.9253 | 206.1 s | 0.384 ms/window |
| LSTM, untuned (2 layers x 64, last-step, 53k params, best epoch 9 of 19) | 0.8616 | 0.8606 | 202.9 s | 0.404 ms/window |
| **LSTM, tuned** (bidirectional, mean-pooled, dropout 0.5, 139k params, best epoch 14 of 24) | 0.8948 | 0.8947 | 912.3 s | 1.464 ms/window |

Random Forest errors stay almost entirely within a group: sitting vs standing (128) and
among the three walking types (138), with no static vs dynamic confusions. The CNN
almost removes the walking-type errors (35) but confuses sitting and standing more often (156),
and makes 30 static vs dynamic errors (walking upstairs vs sitting) that the RF never makes.
LAYING is perfect for both.

**LSTM tuning.** The untuned LSTM overfit early. Four alternatives were scored on the
validation subjects only (`artifacts/tuning/lstm_s42.json`); the test set was not used:

| Candidate | Val macro F1 |
|---|---|
| bidirectional + mean pooling + dropout 0.5 (selected) | 0.9084 |
| mean pooling, lr 5e-4 | 0.9032 |
| mean pooling | 0.9023 |
| bidirectional + mean pooling | 0.8945 |
| untuned default (last-step pooling) | 0.8790 |

Averaging the LSTM outputs over time instead of using only the last step gave most
of the gain. The top three candidates are within 0.006 of each other, which is about
the size of seed-to-seed noise. The tuned LSTM improves test accuracy from 0.862 to
0.895 and cuts static vs dynamic errors from 120 to 69. It is still behind the CNN, and it
is the slowest model (912 s to train, 1.46 ms/window).
CNN and LSTM training time includes the per-epoch validation passes.
These are single-seed numbers; on validation the RF scored slightly higher (macro F1 0.920 vs
0.917), so the ranking is not settled until the multi-seed comparison (mean ± std) in Stage 7.
