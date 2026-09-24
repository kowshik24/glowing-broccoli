"""
Score new machine-sensor readings with the trained failure model.

Loads the artifacts saved by the notebook (models/preprocessor.joblib,
models/xgboost_tuned.joblib, models/metadata.joblib) and applies the same
feature engineering used during training before predicting.

Usage:
    python src/predict.py --csv path/to/new_readings.csv
    python src/predict.py --csv path/to/new_readings.csv --out predictions.csv

Input CSV needs these raw columns (same as the training data, minus the
labels): Type, Air temperature [K], Process temperature [K],
Rotational speed [rpm], Torque [Nm], Tool wear [min]
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def add_engineered_features(df, strain_limit):
    df = df.copy()
    df["Power [W]"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)
    df["Temp_diff [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Overstrain [minNm]"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    df["Overstrain_ratio"] = df["Overstrain [minNm]"] / df["Type"].map(strain_limit)
    return df


def load_artifacts():
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(MODELS_DIR / "xgboost_tuned.joblib")
    metadata = joblib.load(MODELS_DIR / "metadata.joblib")
    return preprocessor, model, metadata


def predict(df_raw, preprocessor, model, metadata):
    df_fe = add_engineered_features(df_raw, metadata["strain_limit"])
    X = df_fe[metadata["feature_cols"]]
    X_proc = preprocessor.transform(X)

    proba = model.predict_proba(X_proc)[:, 1]
    pred = (proba >= metadata["threshold"]).astype(int)

    result = df_raw.copy()
    result["failure_probability"] = proba
    result["predicted_failure"] = pred
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="CSV of new readings to score")
    parser.add_argument("--out", default=None, help="where to write predictions (defaults to stdout)")
    args = parser.parse_args()

    preprocessor, model, metadata = load_artifacts()
    df_raw = pd.read_csv(args.csv)
    result = predict(df_raw, preprocessor, model, metadata)

    if args.out:
        result.to_csv(args.out, index=False)
        print(f"wrote {len(result)} predictions to {args.out}")
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
