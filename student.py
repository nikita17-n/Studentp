
from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, Binarizer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier


@dataclass(frozen=True)
class Config:
    DATA_PATH: Path = Path("student_performance.csv")
    TARGET_COLUMN: str = "grade"

    # ID columns do not help the model learn real patterns.
    DROP_COLUMNS: Tuple[str, ...] = ("student_id",)

    # In your dataset, grade is most likely directly calculated from total_score.
    # Keep this True when you want highest accuracy.
    # Set this False when you want to predict grade without using final score.
    INCLUDE_TOTAL_SCORE: bool = True

    TEST_SIZE: float = 0.20
    RANDOM_STATE: int = 42

    REPORTS_DIR: Path = Path("reports")
    MODELS_DIR: Path = Path("models")

    BEST_MODEL_FILE: str = "best_student_grade_model.joblib"
    MODEL_COMPARISON_FILE: str = "model_comparison.csv"
    CLASSIFICATION_REPORT_FILE: str = "best_classification_report.txt"
    CONFUSION_MATRIX_FILE: str = "best_confusion_matrix.csv"
    METADATA_FILE: str = "training_metadata.json"

    # Heavy models can become slow on 10 lakh rows.
    ENABLE_KNN: bool = False
    KNN_TRAIN_SAMPLE_SIZE: int = 50_000
    KNN_TEST_SAMPLE_SIZE: int = 20_000


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def make_one_hot_encoder() -> OneHotEncoder:
    """
    Makes code compatible with both newer and older scikit-learn versions.
    New versions use sparse_output=False.
    Older versions use sparse=False.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data(config: Config) -> pd.DataFrame:
    if not config.DATA_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found: {config.DATA_PATH.resolve()}\n"
            "Place student_performance.csv in the same folder as this Python file."
        )

    df = pd.read_csv(config.DATA_PATH)

    if df.empty:
        raise ValueError("The CSV file is empty.")

    if config.TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{config.TARGET_COLUMN}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    return df


def clean_data(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    df = df.copy()

    # Remove duplicate rows.
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        logging.info("Removed %s duplicate rows.", before - after)

    # Remove rows where target is missing.
    df = df.dropna(subset=[config.TARGET_COLUMN])

    # Drop ID or unwanted columns safely.
    drop_columns = [col for col in config.DROP_COLUMNS if col in df.columns]

    if not config.INCLUDE_TOTAL_SCORE and "total_score" in df.columns:
        drop_columns.append("total_score")

    if drop_columns:
        df = df.drop(columns=drop_columns)

    if len(df) < 10:
        raise ValueError("Not enough rows after cleaning. At least 10 rows are recommended.")

    return df


def split_features_target(
    df: pd.DataFrame,
    config: Config,
) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    X = df.drop(columns=[config.TARGET_COLUMN])
    target = df[config.TARGET_COLUMN].astype(str)

    if X.empty:
        raise ValueError("No feature columns available for training.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(target)

    logging.info("Features used: %s", list(X.columns))
    logging.info("Target classes: %s", list(label_encoder.classes_))

    return X, y, label_encoder


def build_preprocessor(
    X: pd.DataFrame,
    *,
    scale_numeric: bool,
    binarize_numeric: bool = False,
) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [col for col in X.columns if col not in numeric_columns]

    numeric_steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    if binarize_numeric:
        numeric_steps.append(("binarizer", Binarizer(threshold=0.0)))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    transformers = []

    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_models(X: pd.DataFrame, config: Config) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X, scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=10_000,
                        class_weight="balanced",
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X, scale_numeric=False)),
                (
                    "model",
                    DecisionTreeClassifier(
                        random_state=config.RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X, scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=config.RANDOM_STATE,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Hist Gradient Boosting": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X, scale_numeric=False)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Gaussian Naive Bayes": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X, scale_numeric=False)),
                ("model", GaussianNB()),
            ]
        ),
        "Bernoulli Naive Bayes": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X, scale_numeric=True, binarize_numeric=True)),
                ("model", BernoulliNB()),
            ]
        ),
    }

    if config.ENABLE_KNN:
        models["KNN"] = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X, scale_numeric=True)),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]
        )

    return models


def stratified_sample(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_size: int,
    random_state: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    if len(X) <= sample_size:
        return X, y

    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=sample_size,
        random_state=random_state,
        stratify=y,
    )

    return X_sample, y_sample


def evaluate_model(
    name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    config: Config,
) -> Tuple[Dict[str, Any], np.ndarray, Pipeline]:
    logging.info("Training model: %s", name)

    # KNN is expensive on very large datasets, so sample it safely.
    if name == "KNN":
        X_train_used, y_train_used = stratified_sample(
            X_train,
            y_train,
            config.KNN_TRAIN_SAMPLE_SIZE,
            config.RANDOM_STATE,
        )
        X_test_used, y_test_used = stratified_sample(
            X_test,
            y_test,
            config.KNN_TEST_SAMPLE_SIZE,
            config.RANDOM_STATE,
        )
    else:
        X_train_used, y_train_used = X_train, y_train
        X_test_used, y_test_used = X_test, y_test

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_train_used, y_train_used)

    y_pred = model.predict(X_test_used)

    result = {
        "model": name,
        "train_rows": len(X_train_used),
        "test_rows": len(X_test_used),
        "accuracy": accuracy_score(y_test_used, y_pred),
        "precision_weighted": precision_score(y_test_used, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test_used, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test_used, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_test_used, y_pred, average="macro", zero_division=0),
    }

    logging.info(
        "%s | Accuracy: %.4f | Weighted F1: %.4f | Macro F1: %.4f",
        name,
        result["accuracy"],
        result["f1_weighted"],
        result["f1_macro"],
    )

    return result, y_pred, model


def save_outputs(
    results_df: pd.DataFrame,
    best_model_name: str,
    best_model: Pipeline,
    best_predictions: np.ndarray,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    feature_columns: List[str],
    config: Config,
) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    comparison_path = config.REPORTS_DIR / config.MODEL_COMPARISON_FILE
    results_df.to_csv(comparison_path, index=False)

    target_names = [str(cls) for cls in label_encoder.classes_]

    report_text = classification_report(
        y_test,
        best_predictions,
        labels=np.arange(len(target_names)),
        target_names=target_names,
        zero_division=0,
    )

    report_path = config.REPORTS_DIR / config.CLASSIFICATION_REPORT_FILE
    report_path.write_text(report_text, encoding="utf-8")

    cm = confusion_matrix(y_test, best_predictions, labels=np.arange(len(target_names)))
    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{name}" for name in target_names],
        columns=[f"predicted_{name}" for name in target_names],
    )
    cm_path = config.REPORTS_DIR / config.CONFUSION_MATRIX_FILE
    cm_df.to_csv(cm_path)

    artifact = {
        "model": best_model,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "target_column": config.TARGET_COLUMN,
        "best_model_name": best_model_name,
        "include_total_score": config.INCLUDE_TOTAL_SCORE,
    }

    model_path = config.MODELS_DIR / config.BEST_MODEL_FILE
    joblib.dump(artifact, model_path)

    metadata = {
        "best_model": best_model_name,
        "feature_columns": feature_columns,
        "target_classes": target_names,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "include_total_score": config.INCLUDE_TOTAL_SCORE,
        "saved_files": {
            "best_model": str(model_path),
            "model_comparison": str(comparison_path),
            "classification_report": str(report_path),
            "confusion_matrix": str(cm_path),
        },
    }

    metadata_path = config.REPORTS_DIR / config.METADATA_FILE
    metadata_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")

    logging.info("Saved best model: %s", model_path)
    logging.info("Saved model comparison: %s", comparison_path)
    logging.info("Saved classification report: %s", report_path)
    logging.info("Saved confusion matrix: %s", cm_path)
    logging.info("Saved metadata: %s", metadata_path)


def predict_with_saved_model(input_data: Dict[str, Any], model_file: Path | str = "models/best_student_grade_model.joblib") -> str:
    """
    Example:
        prediction = predict_with_saved_model({
            "weekly_self_study_hours": 18.5,
            "attendance_percentage": 92.0,
            "class_participation": 8.5,
            "total_score": 89.0
        })
        print(prediction)
    """
    artifact = joblib.load(model_file)

    model: Pipeline = artifact["model"]
    label_encoder: LabelEncoder = artifact["label_encoder"]
    feature_columns: List[str] = artifact["feature_columns"]

    missing_columns = [col for col in feature_columns if col not in input_data]
    if missing_columns:
        raise ValueError(f"Missing input columns: {missing_columns}")

    input_df = pd.DataFrame([[input_data[col] for col in feature_columns]], columns=feature_columns)

    encoded_prediction = model.predict(input_df)
    decoded_prediction = label_encoder.inverse_transform(encoded_prediction)

    return str(decoded_prediction[0])


def main() -> None:
    setup_logging()

    config = Config()

    logging.info("Loading dataset...")
    raw_df = load_data(config)

    logging.info("Dataset shape before cleaning: %s", raw_df.shape)
    df = clean_data(raw_df, config)
    logging.info("Dataset shape after cleaning: %s", df.shape)

    X, y, label_encoder = split_features_target(df, config)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    logging.info("Training rows: %s", len(X_train))
    logging.info("Testing rows: %s", len(X_test))

    models = build_models(X, config)

    results = []
    trained_models: Dict[str, Pipeline] = {}
    predictions: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        try:
            result, y_pred, trained_model = evaluate_model(
                name,
                model,
                X_train,
                X_test,
                y_train,
                y_test,
                label_encoder,
                config,
            )
            results.append(result)
            trained_models[name] = trained_model
            predictions[name] = y_pred
        except Exception as exc:
            logging.exception("Model failed: %s | Error: %s", name, exc)

    if not results:
        raise RuntimeError("No model trained successfully.")

    results_df = pd.DataFrame(results).sort_values(
        by=["f1_weighted", "accuracy"],
        ascending=False,
    )

    print("\n================ MODEL COMPARISON ================\n")
    print(results_df.to_string(index=False))

    best_model_name = str(results_df.iloc[0]["model"])
    best_model = trained_models[best_model_name]
    best_predictions = predictions[best_model_name]

    print("\n================ BEST MODEL ================\n")
    print(f"Best Model: {best_model_name}")
    print(f"Accuracy: {results_df.iloc[0]['accuracy']:.4f}")
    print(f"Weighted F1: {results_df.iloc[0]['f1_weighted']:.4f}")
    print(f"Macro F1: {results_df.iloc[0]['f1_macro']:.4f}")

    save_outputs(
        results_df=results_df,
        best_model_name=best_model_name,
        best_model=best_model,
        best_predictions=best_predictions,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        label_encoder=label_encoder,
        feature_columns=list(X.columns),
        config=config,
    )

    print("\nTraining completed successfully.")
    print("Check the 'reports' folder for metrics.")
    print("Check the 'models' folder for the saved best model.")


if __name__ == "__main__":
    main()
