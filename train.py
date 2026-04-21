"""Train a RandomForest model on wine quality data and log to MLflow."""
from __future__ import annotations
import os
import sys
import argparse
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

from utils import load_data, features_and_target


def parse_args():
    p = argparse.ArgumentParser("Simple MLflow demo (wine quality prediction)")
    p.add_argument("--csv", default="data/wine_sample.csv",
                   help="Path to CSV (default: data/wine_sample.csv)")
    p.add_argument("--target", default="quality",
                   help="Target column name (default: quality)")
    p.add_argument("--experiment", default="wine-prediction",
                   help="MLflow experiment name")
    p.add_argument("--run", default="run-1",
                   help="MLflow run name")
    p.add_argument("--n-estimators", type=int, default=50,
                   help="RandomForest n_estimators (default: 50)")
    p.add_argument("--max-depth", type=int, default=5,
                   help="RandomForest max_depth (default: 5)")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Test split fraction (default: 0.2)")
    p.add_argument("--random-state", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--log-model", action="store_true",
                   help="Also log the trained model to MLflow (requires artifact store)")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- MLflow setup ----
    # Reads from env var first; fall back to localhost:5000 (standard MLflow port)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    print(f"[INFO] Tracking URI: {tracking_uri}")
    print(f"[INFO] Experiment:   {args.experiment}")
    print(f"[INFO] Run name:     {args.run}")

    # ---- Load data ----
    if not os.path.exists(args.csv):
        sys.exit(f"[ERROR] CSV not found: {args.csv}")

    df = load_data(args.csv)
    print(f"[INFO] Loaded {len(df)} rows from {args.csv}")

    if args.target not in df.columns:
        sys.exit(f"[ERROR] Target column '{args.target}' not found. "
                 f"Columns: {list(df.columns)}")

    X, y = features_and_target(df, target=args.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    print(f"[INFO] Train rows: {len(X_train)}, Test rows: {len(X_test)}")

    # ---- Train and log ----
    with mlflow.start_run(run_name=args.run) as run:
        # Log params
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        # Train
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
        )
        model.fit(X_train, y_train)

        # Evaluate
        preds = model.predict(X_test)
        mse = float(mean_squared_error(y_test, preds))
        rmse = float(math.sqrt(mse))
        r2 = float(r2_score(y_test, preds))

        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Optionally log the trained model (uploads to your S3 artifact store)
        if args.log_model:
            mlflow.sklearn.log_model(model, name="model")
            print("[INFO] Model logged to MLflow artifact store.")

        print(f"[RESULT] MSE : {mse:.4f}")
        print(f"[RESULT] RMSE: {rmse:.4f}")
        print(f"[RESULT] R2  : {r2:.4f}")
        print(f"[DONE]  Run ID: {run.info.run_id}")
        print(f"[DONE]  View in UI: {tracking_uri}/#/experiments/"
              f"{run.info.experiment_id}/runs/{run.info.run_id}")


if __name__ == "__main__":
    main()