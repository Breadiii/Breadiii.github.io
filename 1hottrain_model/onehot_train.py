from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_CANDIDATES = [
    BASE_DIR / "backend" / "Public_services.csv",
    Path(__file__).resolve().parent / "public_service.csv",
]
MODEL_PATH = BASE_DIR / "backend" / "pressure_model.pkl"
TARGET = "PRESSURE_SCORE_GAUSSIAN"
CAT_FEATURES = [
    "LOCATION_POSTAL_CODE",
    "SECTOR",
    "OVERNIGHT_SERVICE_TYPE",
    "PROGRAM_MODEL",
    "PROGRAM_AREA",
    "CAPACITY_TYPE",
]
NUM_FEATURES = [
    "ACTUAL_CAPACITY",
    "lat",
    "lon",
    "dow",
    "month",
    "day",
]


def build_one_hot_encoder() -> OneHotEncoder:
    version_match = re.match(r"^(\d+)\.(\d+)", sklearn_version)
    if version_match:
        sk_major, sk_minor = map(int, version_match.groups())
        if (sk_major, sk_minor) >= (1, 2):
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", build_one_hot_encoder()),
                    ]
                ),
                CAT_FEATURES,
            ),
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median"))]),
                NUM_FEATURES,
            ),
        ]
    )

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=400,
        random_state=42,
    )
    return Pipeline([("prep", preprocess), ("model", model)])


def resolve_data_path() -> Path:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    searched = ", ".join(str(path) for path in DATA_CANDIDATES)
    raise FileNotFoundError(f"No training CSV found. Checked: {searched}")


def load_training_data() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Index, pd.Index]:
    df = pd.read_csv(resolve_data_path())
    df = df.dropna(subset=[TARGET]).copy()

    df["OCCUPANCY_DATE"] = pd.to_datetime(df["OCCUPANCY_DATE"], errors="coerce")
    df["dow"] = df["OCCUPANCY_DATE"].dt.dayofweek
    df["month"] = df["OCCUPANCY_DATE"].dt.month
    df["day"] = df["OCCUPANCY_DATE"].dt.day

    x = df[CAT_FEATURES + NUM_FEATURES]
    eps = 1e-6
    y_raw = df[TARGET].clip(eps, 1 - eps)
    y = np.log(y_raw / (1 - y_raw))

    df_sorted = df.sort_values("OCCUPANCY_DATE")
    cut = int(len(df_sorted) * 0.8)
    train_idx = df_sorted.index[:cut]
    test_idx = df_sorted.index[cut:]
    return x, y, y_raw, train_idx, test_idx


def save_artifact(pipe: Pipeline) -> None:
    artifact = {
        "model": pipe,
        "target": TARGET,
        "cat_features": CAT_FEATURES,
        "num_features": NUM_FEATURES,
    }
    with MODEL_PATH.open("wb") as fh:
        pickle.dump(artifact, fh)


def main() -> None:
    x, y, y_raw, train_idx, test_idx = load_training_data()
    x_train, y_train = x.loc[train_idx], y.loc[train_idx]
    x_test, y_test = x.loc[test_idx], y.loc[test_idx]

    pipe = build_pipeline()
    pipe.fit(x_train, y_train)
    save_artifact(pipe)

    pred_logit = pipe.predict(x_test)
    pred = 1 / (1 + np.exp(-pred_logit))

    mae = mean_absolute_error(y_raw.loc[test_idx], pred)
    rmse = mean_squared_error(y_raw.loc[test_idx], pred) ** 0.5
    naive_pred = np.full(len(y_test), y_raw.loc[train_idx].mean())

    print("Model saved to:", MODEL_PATH)
    print("Baseline MAE:", mean_absolute_error(y_raw.loc[test_idx], naive_pred))
    print(y_raw.describe())
    print(df_corr(y_raw))
    print("MAE:", mae)
    print("RMSE:", rmse)


def df_corr(y_raw: pd.Series) -> pd.Series:
    df = pd.read_csv(resolve_data_path())
    df["OCCUPANCY_DATE"] = pd.to_datetime(df["OCCUPANCY_DATE"], errors="coerce")
    df["dow"] = df["OCCUPANCY_DATE"].dt.dayofweek
    df["month"] = df["OCCUPANCY_DATE"].dt.month
    df["day"] = df["OCCUPANCY_DATE"].dt.day
    return df[NUM_FEATURES + [TARGET]].corr()[TARGET].sort_values()


if __name__ == "__main__":
    main()
