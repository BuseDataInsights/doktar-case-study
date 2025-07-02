# VMC Sensor Calibration Pipeline

This repository contains production-ready code to learn a mapping from normalized soil-moisture sensor counts to volumetric moisture content (VMC %).

## Features

- **Multiple ML Models**: Linear regression, polynomial features, isotonic regression, and gradient boosting
- **Robust Validation**: 5-fold cross-validation with RMSE, MAE, and R² metrics
- **Production Ready**: Containerized deployment with Docker
- **CLI Interface**: Easy-to-use command-line tools for training and prediction
- **Comprehensive Testing**: Unit tests and integration tests included

## Performance

The pipeline automatically selects the best performing model based on cross-validated RMSE:

| Model        | CV-RMSE  | CV-MAE   | CV-R²    |
| ------------ | -------- | -------- | -------- |
| **Isotonic** | **3.94** | **3.07** | **0.89** |
| Polynomial   | 4.02     | 3.09     | 0.88     |
| GBDT         | 3.96     | 3.08     | 0.89     |
| Linear       | 4.91     | 3.76     | 0.82     |

## Installation

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/BuseDataInsights/doktar-case-study.git
cd q1

# Build Docker image
docker build -t vmc_calib .
```

### Option 2: Local Installation

```bash
# Clone and setup virtual environment
git clone https://github.com/BuseDataInsights/doktar-case-study.git
cd q1
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Quick Start

### Training

```bash
# Docker
docker run --rm -v $(pwd)/data:/app/data vmc_calib train

# Local
python -m vmc_model.cli train
```

### Prediction

```bash
# Docker
docker run --rm -v $(pwd)/data:/app/data -v $(pwd):/app/output \
  vmc_calib predict --source data/your_data.xlsx --output /app/output/predictions.csv

# Local
python -m vmc_model.cli predict --source data/your_data.xlsx
```

## Data Format

Input data should be Excel (.xlsx) or CSV files with columns:

- `Normalized Values`: Sensor readings (normalized counts)
- `Measured VMC (%)`: Ground truth volumetric moisture content (%)

## Architecture

```
vmc_model/
├── __init__.py          # Package initialization
├── cli.py              # Command-line interface
├── data.py             # Data loading and preprocessing
├── models.py           # Model definitions and evaluation
├── train.py            # Training pipeline
└── predict.py          # Prediction pipeline
```

## Testing

```bash
# Run tests
python -m pytest tests/

# With coverage
python -m pytest tests/ --cov=vmc_model
```

## Model Selection

The pipeline evaluates four candidate models:

1. **Linear Regression**: Simple baseline
2. **Polynomial Features (degree=2)**: Captures nonlinear relationships
3. **Isotonic Regression**: Enforces monotonicity constraint
4. **Gradient Boosting**: Flexible ensemble method with monotonic constraints

Selection is based on 5-fold cross-validated RMSE, with isotonic regression typically performing best due to its physical monotonicity constraint.

## Configuration

Model hyperparameters are configured in `config.yaml`:

```yaml
models:
  gbdt:
    n_estimators: 300
    learning_rate: 0.05
    max_depth: 3
  isotonic:
    out_of_bounds: clip
```
