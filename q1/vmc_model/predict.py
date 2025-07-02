"""Batch prediction CLI: read a file with Normalized_Values, append predicted VMC."""
from pathlib import Path
import pandas as pd
import joblib
import yaml


def main(source: str, model: str = "model.joblib", output: str = "predictions.csv"):
    """Predict VMC from normalized sensor readings."""
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    feature_col = cfg["feature_col"]

    if source.endswith((".xlsx", ".xls")):
        df = pd.read_excel(source, engine="openpyxl")
    else:
        df = pd.read_csv(source)

    model_obj = joblib.load(model)
    preds = model_obj.predict(df[[feature_col]].values.ravel() if hasattr(model_obj, "X_min_") else df[[feature_col]].values)
    df["Predicted_VMC"] = preds
    df.to_csv(output, index=False)
    print(f"Wrote {len(df)} rows to {output}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict VMC from normalized sensor readings.")
    parser.add_argument("--model", default="model.joblib", help="Path to the trained .joblib model")
    parser.add_argument("--source", required=True, help="CSV / Excel file with Normalized_Values column")
    parser.add_argument("--output", default="predictions.csv", help="Destination CSV file")
    args = parser.parse_args()
    main(source=args.source, model=args.model, output=args.output)
