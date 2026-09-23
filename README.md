# Sales Regression Analysis

A small machine-learning analysis that preprocesses retail sales data and compares simple linear, multiple linear, polynomial, Ridge, and Lasso regression models.

## Features

- Median imputation for missing numerical values
- Categorical encoding and feature standardization
- Sales prediction for a new store
- Actual-versus-predicted plots
- Train/test evaluation using RMSE, MAE, and R²

## Run locally

```bash
python -m venv .venv
```

Activate the virtual environment, then install the dependencies and run the analysis:

```bash
pip install -r requirements.txt
python app.py
```

The script prints the cleaned dataset, new-store predictions, and model evaluation metrics, and displays three comparison plots.
