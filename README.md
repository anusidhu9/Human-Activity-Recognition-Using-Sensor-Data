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
python scripts/train.py --model lstm --seed 42
```

## Results so far

Single run, seed 42. Trained on 5,952 windows from 17 subjects, model selected on
4 held-out training subjects (3, 16, 22, 23), evaluated once on the official test set
(2,947 windows, 9 unseen subjects). CPU, 4 threads.

| Model | Test accuracy | Macro F1 | Train time | Inference |
|---|---|---|---|---|
| Random Forest (300 trees, max_features=sqrt, min_samples_leaf=5) | 0.9097 | 0.9073 | 14.2 s | 0.063 ms/window |
| 1D CNN (3 conv blocks, 127k params, best epoch 23 of 33) | 0.9250 | 0.9253 | 206.1 s | 0.384 ms/window |
| LSTM (2 layers x 64, 53k params, best epoch 9 of 19) | 0.8616 | 0.8606 | 202.9 s | 0.404 ms/window |

Random Forest errors stay almost entirely within a group: sitting vs standing (128) and
among the three walking types (138), with no static vs dynamic confusions. The CNN
almost removes the walking-type errors (35) but confuses sitting and standing more often (156),
and makes 30 static vs dynamic errors (walking upstairs vs sitting) that the RF never makes.
LAYING is perfect for both. The LSTM (default settings, untuned) is clearly behind: it overfits
early (validation loss rises from epoch 3 while training loss keeps falling) and makes
120 static vs dynamic errors, e.g. standing predicted as walking (40) and laying predicted as
walking upstairs (27). CNN and LSTM training time includes the per-epoch validation passes.
These are single-seed numbers; on validation the RF scored slightly higher (macro F1 0.920 vs
0.917), so the ranking is not settled until the multi-seed comparison (mean ± std) in Stage 7.
