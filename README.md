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

> **Status:** work in progress, built in stages. Stages 0-2 (environment, project
> skeleton, dataset download and inspection) are done. Results will be added once the models
> have actually been trained.

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
```
