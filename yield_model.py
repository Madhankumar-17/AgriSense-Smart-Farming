
"""Train the AgriSense crop yield prediction model."""

from pathlib import Path
import sys
import pickle

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "yield_model.pkl"

CATEGORICAL = ["crop", "state", "season"]

NUMERIC = [
    "rainfall_mm",
    "temperature_c",
    "soil_ph",
    "nitrogen_kg_ha",
    "phosphorus_kg_ha",
    "potassium_kg_ha",
]

FEATURES = CATEGORICAL + NUMERIC

TARGET = "yield_tonnes_per_hectare"


def main():

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python train_yield_model.py "
            "data/crop_yield_dataset.csv"
        )

    csv_path = Path(sys.argv[1])

    if not csv_path.is_absolute():
        csv_path = BASE_DIR / csv_path

    if not csv_path.is_file():
        raise SystemExit(
            f"Dataset not found: {csv_path}"
        )

    print("Reading crop dataset...")

    raw = pd.read_csv(csv_path)

    required = [
        "Region",
        "Crop",
        "Season",
        "Rainfall",
        "Temperature",
        "pH",
        "Nitrogen",
        "Phosphorus",
        "Potassium",
        "Yield"
    ]

    missing = [
        col for col in required
        if col not in raw.columns
    ]

    if missing:
        raise SystemExit(
            f"Missing dataset columns: {missing}"
        )

    data = pd.DataFrame({

        "crop": raw["Crop"].astype("string").str.strip(),

        "state": (
            raw["Region"]
            .astype("string")
            .str.split(",")
            .str[-1]
            .str.strip()
        ),

        "season": (
            raw["Season"]
            .astype("string")
            .str.strip()
        ),

        "rainfall_mm": raw["Rainfall"],

        "temperature_c": raw["Temperature"],

        "soil_ph": raw["pH"],

        "nitrogen_kg_ha": raw["Nitrogen"],

        "phosphorus_kg_ha": raw["Phosphorus"],

        "potassium_kg_ha": raw["Potassium"],

        TARGET: raw["Yield"]

    })

    # Convert numerical columns.

    for col in NUMERIC + [TARGET]:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    # Convert kg/ha into tonnes/ha.
    # Verify that the original Yield column is in kg/ha.

    data[TARGET] = data[TARGET] / 1000.0

    data = data.dropna(subset=[TARGET])

    data = data[data[TARGET] >= 0]

    data = data.dropna(subset=CATEGORICAL)

    if len(data) < 20:
        raise SystemExit(
            "Not enough valid rows to train."
        )

    X = data[FEATURES]
    y = data[TARGET]

    preprocessor = ColumnTransformer([

        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]),
            CATEGORICAL
        ),

        (
            "numeric",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scale",
                    StandardScaler()
                )
            ]),
            NUMERIC
        )

    ])

    pipeline = Pipeline([

        ("preprocessor", preprocessor),

        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )

    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(f"Training rows: {len(X_train)}")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    print("\nMODEL EVALUATION")

    print(f"Valid rows: {len(data)}")

    print(
        "MAE:",
        round(
            mean_absolute_error(y_test, predictions),
            4
        ),
        "tonnes/ha"
    )

    print(
        "RMSE:",
        round(
            mean_squared_error(y_test, predictions) ** 0.5,
            4
        ),
        "tonnes/ha"
    )

    print(
        "R2 Score:",
        round(
            r2_score(y_test, predictions),
            4
        )
    )

    # Save trained model.

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with MODEL_PATH.open("wb") as file:
        pickle.dump(pipeline, file)

    print("\nMODEL SAVED SUCCESSFULLY")

    print(MODEL_PATH)

    print("\nExpected input features:")

    print(FEATURES)


if __name__ == "__main__":
    main()
