#!/usr/bin/env python3
"""Revision pipeline for the Scientific Reports resubmission.

This script migrates the experiment logic used by the notebooks into a
repeatable CLI. It intentionally writes experiment artifacts under
``experiments/*/outputs`` and ``experiments/revision_outputs`` only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

SEED = 23
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
REVISION_OUTPUTS_DIR = EXPERIMENTS_DIR / "revision_outputs"


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    cli_name: str
    short: str
    directory: Path
    input_csv: str
    target_col: str
    label_name: str
    categorical_cols: tuple[str, ...]
    age_mu: int
    age_sigma: int
    protected_region_codes: frozenset[int]
    model_family: str


DATASETS = {
    "drug-trafficking": DatasetConfig(
        key="drug_trafficking",
        cli_name="drug-trafficking",
        short="dt",
        directory=EXPERIMENTS_DIR / "drugTrafficking",
        input_csv="drug_traffiking_anon.csv",
        target_col="Tipo salida 2",
        label_name="label_value",
        categorical_cols=("region", "barrister", "crime_stage", "foreigner"),
        age_mu=26,
        age_sigma=8,
        protected_region_codes=frozenset({13, 5, 0, 1, 2, 11, 15, 4}),
        model_family="keras",
    ),
    "petty-theft": DatasetConfig(
        key="petty_theft",
        cli_name="petty-theft",
        short="pt",
        directory=EXPERIMENTS_DIR / "pettyTheft",
        input_csv="petty_theft_anon.csv",
        target_col="Tipo salida 1",
        label_name="label_value",
        categorical_cols=("region", "barrister", "crime_stage"),
        age_mu=31,
        age_sigma=8,
        protected_region_codes=frozenset({14, 12, 11, 8, 7, 6, 5, 4}),
        model_family="lightgbm",
    ),
}

REFERENCE_CONFUSIONS = {
    "drug_trafficking": {
        "base": {"tp": 1594, "tn": 2323, "fp": 3, "fn": 1087},
        "improved": {"tp": 1833, "tn": 1413, "fp": 913, "fn": 848},
    },
    "petty_theft": {
        "base": {"tp": 878, "tn": 8279, "fp": 432, "fn": 2398},
        "improved": {"tp": 1174, "tn": 7647, "fp": 1064, "fn": 2102},
    },
}


MODEL_LABELS = ("base", "improved")
AGE_DISTRIBUTIONS = ("normal", "uniform")


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def minmax_scale(train: pd.DataFrame, test: pd.DataFrame, val: pd.DataFrame):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns)
    val_scaled = pd.DataFrame(scaler.transform(val), columns=val.columns)
    return train_scaled, test_scaled, val_scaled


def normalize_region(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["Región (tribunal)"] = data["Región (tribunal)"].replace(
        {"Metropolitana Sur": "Metropolitana", "Metropolitana Norte": "Metropolitana"}
    )
    return data


def add_synthetic_age(
    data: pd.DataFrame, config: DatasetConfig, age_distribution: str
) -> pd.DataFrame:
    data = data.copy()
    if age_distribution == "normal":
        age_values = np.random.normal(config.age_mu, config.age_sigma, len(data))
        age_values = np.clip(age_values, 18, 65).round().astype(int)
    elif age_distribution == "uniform":
        age_values = np.random.uniform(18, 65, len(data)).round().astype(int)
    else:
        raise ValueError(f"Unsupported age distribution: {age_distribution}")

    data[f"age_{age_distribution}"] = age_values
    return data


def load_dataset(config: DatasetConfig, age_distribution: str) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(config.directory / config.input_csv)
    data = normalize_region(data)
    data = add_synthetic_age(data, config, age_distribution)

    if config.cli_name == "drug-trafficking":
        data = data.drop(index=[16690, 16691], errors="ignore")
        data["foreigner"] = np.where(data["P.S. Expulsión"] == "N", "no", "yes")
        data = data.drop(
            data.columns[[2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]],
            axis=1,
        )
    else:
        data = data.drop(data.columns[[1, 3, 4, 5, 6, 8, 10, 11, 12]], axis=1)

    data = data.rename(
        columns={
            "Región (tribunal)": "region",
            "Grado desarrollo": "crime_stage",
            "Defensor": "barrister",
            "Audiencias efectivas": "effective_hearings",
        }
    )
    data = data.reset_index(names="row_id")
    y = data[config.target_col].copy()
    x = data.drop(columns=[col for col in ("Tipo salida 1", "Tipo salida 2") if col in data])
    return x, y


def prepare_splits(config: DatasetConfig, age_distribution: str):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

    x, y = load_dataset(config, age_distribution)
    x_train, x_test, y_train_raw, y_test_raw = train_test_split(
        x, y, test_size=0.3, random_state=SEED
    )
    x_val = x_train.tail(1000).copy()
    y_val_raw = y_train_raw.tail(1000).copy()

    categorical_cols = list(config.categorical_cols)
    numeric_cols = [col for col in x_train.columns if col not in categorical_cols + ["row_id"]]

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_train = pd.DataFrame(
        encoder.fit_transform(x_train[categorical_cols]),
        columns=categorical_cols,
        index=x_train.index,
    ).astype(int)
    cat_test = pd.DataFrame(
        encoder.transform(x_test[categorical_cols]),
        columns=categorical_cols,
        index=x_test.index,
    ).astype(int)
    cat_val = pd.DataFrame(
        encoder.transform(x_val[categorical_cols]),
        columns=categorical_cols,
        index=x_val.index,
    ).astype(int)

    num_train, num_test, num_val = minmax_scale(
        x_train[numeric_cols], x_test[numeric_cols], x_val[numeric_cols]
    )
    num_train.index = x_train.index
    num_test.index = x_test.index
    num_val.index = x_val.index

    x_train_model = pd.concat([num_train, cat_train], axis=1).reset_index(drop=True)
    x_test_model = pd.concat([num_test, cat_test], axis=1).reset_index(drop=True)
    x_val_model = pd.concat([num_val, cat_val], axis=1).reset_index(drop=True)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)
    y_val = label_encoder.transform(y_val_raw)

    audit_train = x_train.reset_index(drop=True)
    audit_test = x_test.reset_index(drop=True)
    audit_val = x_val.reset_index(drop=True)
    return {
        "x_train": x_train_model,
        "x_test": x_test_model,
        "x_val": x_val_model,
        "y_train": y_train,
        "y_test": y_test,
        "y_val": y_val,
        "audit_train": audit_train,
        "audit_test": audit_test,
        "audit_val": audit_val,
        "label_classes": list(label_encoder.classes_),
    }


def confusion_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    for_rate = fn / (fn + tn) if (fn + tn) else 0.0
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fpr,
        "fnr": fnr,
        "for": for_rate,
    }


def append_metrics(
    rows: list[dict[str, object]],
    dataset: str,
    model_label: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    row = {"dataset": dataset, "model_label": model_label, "split": split}
    row.update(confusion_dict(y_true, y_pred))
    rows.append(row)


def build_keras_model(num_features: int, improved: bool):
    from tensorflow import keras
    from keras import layers

    hidden_1 = 14 if improved else 13
    hidden_2 = 12 if improved else 6
    activation = "relu" if improved else "sigmoid"
    inputs = keras.Input(shape=(num_features,), name="data")
    x = layers.Dense(hidden_1, activation=activation, name="dense_1")(inputs)
    x = layers.Dense(hidden_2, activation=activation, name="dense_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.BinaryAccuracy(name="binary_accuracy"),
        ],
    )
    return model


def train_keras(config: DatasetConfig, splits: dict, model_label: str):
    improved = model_label == "improved"
    model = build_keras_model(splits["x_train"].shape[1], improved=improved)
    fit_kwargs = {
        "x": splits["x_train"],
        "y": splits["y_train"],
        "batch_size": 1024,
        "epochs": 70,
        "validation_data": (splits["x_val"], splits["y_val"]),
        "verbose": 0,
    }
    if improved:
        fit_kwargs["sample_weight"] = build_reweighing_sample_weight(config, splits)
    model.fit(**fit_kwargs)
    model.classes_ = np.array([0, 1])
    return model


def train_lightgbm(config: DatasetConfig, splits: dict, model_label: str):
    import lightgbm as lgb
    from sklearn.utils import class_weight

    if model_label == "improved":
        weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(splits["y_train"]),
            y=splits["y_train"],
        )
        model = lgb.LGBMClassifier(
            boosting_type="gbdt",
            num_leaves=120,
            max_depth=-1,
            learning_rate=0.01,
            n_estimators=100,
            objective="binary",
            class_weight={0: weights[0], 1: weights[1]},
            random_state=SEED,
            verbose=-1,
        )
        sample_weight = build_reweighing_sample_weight(config, splits)
        model.fit(splits["x_train"], splits["y_train"], sample_weight=sample_weight)
    else:
        model = lgb.LGBMClassifier(
            boosting_type="gbdt",
            num_leaves=63,
            max_depth=-1,
            learning_rate=0.01,
            n_estimators=100,
            objective="binary",
            random_state=SEED,
            verbose=-1,
        )
        model.fit(splits["x_train"], splits["y_train"])
    return model


def build_reweighing_sample_weight(config: DatasetConfig, splits: dict) -> np.ndarray:
    from aif360.algorithms.preprocessing import Reweighing
    from aif360.datasets import BinaryLabelDataset

    train_for_fairness = splits["x_train"].copy()
    train_for_fairness["region_alt"] = train_for_fairness["region"].isin(
        config.protected_region_codes
    ).astype(int)
    train_for_fairness[config.label_name] = splits["y_train"]
    dataset_train = BinaryLabelDataset(
        df=train_for_fairness,
        label_names=[config.label_name],
        protected_attribute_names=["region_alt"],
        favorable_label=1,
        unfavorable_label=0,
    )
    rw = Reweighing(
        unprivileged_groups=[{"region_alt": 0}],
        privileged_groups=[{"region_alt": 1}],
    )
    return rw.fit_transform(dataset_train).instance_weights


def predict_model(model, config: DatasetConfig, x: pd.DataFrame) -> np.ndarray:
    if config.model_family == "keras":
        return (model.predict(x, verbose=0).reshape(-1) > 0.5).astype(int)
    if hasattr(model, "predict_proba"):
        return (model.predict_proba(x)[:, 1] > 0.5).astype(int)
    return model.predict(x).astype(int)


def score_model(model, config: DatasetConfig, x: pd.DataFrame) -> np.ndarray:
    if config.model_family == "keras":
        return model.predict(x, verbose=0).reshape(-1)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.predict(x).astype(float)


def threshold_optimize_predictions(model, config: DatasetConfig, splits: dict):
    from fairlearn.postprocessing import ThresholdOptimizer

    predict_method = "predict" if config.model_family == "keras" else "predict_proba"
    estimator = QuietKerasEstimator(model) if config.model_family == "keras" else model
    optimizer = ThresholdOptimizer(
        estimator=estimator,
        constraints="false_negative_rate_parity",
        objective="accuracy_score",
        prefit=True,
        predict_method=predict_method,
    )
    optimizer.fit(
        splits["x_train"],
        splits["y_train"],
        sensitive_features=splits["x_train"]["region"],
    )
    train_pred = optimizer.predict(
        splits["x_train"], sensitive_features=splits["x_train"]["region"]
    ).astype(int)
    test_pred = optimizer.predict(
        splits["x_test"], sensitive_features=splits["x_test"]["region"]
    ).astype(int)
    return train_pred, test_pred


class QuietKerasEstimator:
    def __init__(self, model):
        self.model = model
        self.classes_ = np.array([0, 1])

    def predict(self, x):
        return self.model.predict(x, verbose=0).reshape(-1)


def write_prediction_csv(
    config: DatasetConfig,
    splits: dict,
    model_label: str,
    age_distribution: str,
    score: np.ndarray,
    y_pred: np.ndarray,
) -> Path:
    output_dir = config.directory / "outputs"
    ensure_dir(output_dir)
    audit = splits["audit_test"].copy()
    out = audit.copy()
    out["score"] = y_pred.astype(int)
    out["prediction_score"] = score
    out["y_true"] = splits["y_test"].astype(int)
    out[config.label_name] = splits["y_test"].astype(int)
    out["model_label"] = model_label
    out["dataset"] = config.key
    out["age_distribution"] = age_distribution
    out["label_classes"] = json.dumps(splits["label_classes"], ensure_ascii=True)
    columns = [
        "row_id",
        *[col for col in audit.columns if col != "row_id"],
        "score",
        "prediction_score",
        "y_true",
        config.label_name,
        "model_label",
        "dataset",
        "age_distribution",
        "label_classes",
    ]
    output_path = output_dir / f"preds_{config.short}_{model_label}_{age_distribution}.csv"
    out[columns].to_csv(output_path, index=False)
    return output_path


def maybe_save_model(model, config: DatasetConfig, model_label: str, age_distribution: str) -> None:
    output_dir = config.directory / "outputs"
    if config.model_family == "keras":
        suffix = f"{config.short}_{model_label}_{age_distribution}.keras"
        model.save(output_dir / suffix)
    else:
        import joblib

        suffix = f"{config.short}_{model_label}_{age_distribution}.pkl"
        joblib.dump(model, output_dir / suffix)


def maybe_write_shap(
    model,
    config: DatasetConfig,
    splits: dict,
    model_label: str,
    age_distribution: str,
    sample_size: int,
) -> None:
    if sample_size == 0:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    output_dir = config.directory / "outputs"
    x_train = splits["x_train"]
    if sample_size > 0:
        x_shap = x_train.head(sample_size)
    else:
        x_shap = x_train
    explainer = shap.Explainer(model, x_shap.values)
    shap_values = explainer(x_shap.values)
    shap_values.feature_names = list(x_shap.columns)
    shap_df = pd.DataFrame(shap_values.values, columns=shap_values.feature_names)
    summary = pd.DataFrame(
        {
            "feature": shap_df.columns,
            "mean_shap": shap_df.mean().values,
            "mean_abs_shap": shap_df.abs().mean().values,
            "min_shap": shap_df.min().values,
            "max_shap": shap_df.max().values,
        }
    )
    summary.to_csv(
        output_dir / f"{config.short}_{model_label}_{age_distribution}_shap_summary.csv",
        index=False,
    )
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{config.short}_{model_label}_{age_distribution}_shap.pdf",
        bbox_inches="tight",
        format="pdf",
    )
    plt.close("all")


def run_models(args: argparse.Namespace) -> None:
    set_global_seed()
    selected = select_datasets(args.dataset)
    performance_rows: list[dict[str, object]] = []
    for config in selected:
        splits = prepare_splits(config, args.age_distribution)
        for model_label in select_model_labels(args.model):
            print(f"Training {config.cli_name} {model_label} ({args.age_distribution})")
            if config.model_family == "keras":
                model = train_keras(config, splits, model_label)
            else:
                model = train_lightgbm(config, splits, model_label)

            train_pred = predict_model(model, config, splits["x_train"])
            test_pred = predict_model(model, config, splits["x_test"])
            if model_label == "improved":
                train_pred, test_pred = threshold_optimize_predictions(model, config, splits)

            score = score_model(model, config, splits["x_test"])
            append_metrics(
                performance_rows,
                config.key,
                model_label,
                "train",
                splits["y_train"],
                train_pred,
            )
            append_metrics(
                performance_rows,
                config.key,
                model_label,
                "test",
                splits["y_test"],
                test_pred,
            )
            path = write_prediction_csv(
                config, splits, model_label, args.age_distribution, score, test_pred
            )
            print(f"Wrote {path.relative_to(REPO_ROOT)}")
            maybe_save_model(model, config, model_label, args.age_distribution)
            if not args.skip_shap:
                maybe_write_shap(
                    model,
                    config,
                    splits,
                    model_label,
                    args.age_distribution,
                    args.shap_sample,
                )

    if performance_rows:
        ensure_dir(REVISION_OUTPUTS_DIR)
        performance = pd.DataFrame(performance_rows)
        performance.to_csv(REVISION_OUTPUTS_DIR / "performance_summary.csv", index=False)
        print(performance.to_string(index=False))


def prediction_path(config: DatasetConfig, model_label: str, age_distribution: str) -> Path:
    return config.directory / "outputs" / f"preds_{config.short}_{model_label}_{age_distribution}.csv"


def load_prediction_pair(config: DatasetConfig, age_distribution: str):
    base_path = prediction_path(config, "base", age_distribution)
    improved_path = prediction_path(config, "improved", age_distribution)
    missing = [str(path) for path in (base_path, improved_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prediction CSVs. Run run-models first. Missing: " + ", ".join(missing)
        )
    base = pd.read_csv(base_path)
    improved = pd.read_csv(improved_path)
    base = base.sort_values("row_id").reset_index(drop=True)
    improved = improved.sort_values("row_id").reset_index(drop=True)
    if not base["row_id"].equals(improved["row_id"]):
        raise ValueError(f"{config.cli_name} base/improved predictions are not pairable by row_id")
    if not base["y_true"].equals(improved["y_true"]):
        raise ValueError(f"{config.cli_name} base/improved y_true values differ")
    return base, improved


def metric_value(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    if metric == "fnr":
        return 1.0 - float(recall_score(y_true, y_pred, zero_division=0))
    if metric == "fpr":
        confusion = confusion_dict(y_true, y_pred)
        tn = confusion["tn"]
        fp = confusion["fp"]
        return fp / (fp + tn) if (fp + tn) else 0.0
    if metric == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    raise ValueError(f"Unsupported metric: {metric}")


def bootstrap_ci(
    y_true: np.ndarray,
    base_pred: np.ndarray,
    improved_pred: np.ndarray,
    metric: str,
    iterations: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    observed = metric_value(y_true, improved_pred, metric) - metric_value(y_true, base_pred, metric)
    diffs = []
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        diffs.append(
            metric_value(y_true[idx], improved_pred[idx], metric)
            - metric_value(y_true[idx], base_pred[idx], metric)
        )
    low, high = np.percentile(diffs, [2.5, 97.5])
    return observed, float(low), float(high)


def run_stats(args: argparse.Namespace) -> None:
    from statsmodels.stats.contingency_tables import mcnemar

    ensure_dir(REVISION_OUTPUTS_DIR)
    paired_rows = []
    perf_rows = []
    reference_rows = []
    for config in select_datasets(args.dataset):
        base, improved = load_prediction_pair(config, args.age_distribution)
        y_true = base["y_true"].to_numpy(dtype=int)
        base_pred = base["score"].to_numpy(dtype=int)
        improved_pred = improved["score"].to_numpy(dtype=int)

        for model_label, pred in (("base", base_pred), ("improved", improved_pred)):
            row = {
                "dataset": config.key,
                "model_label": model_label,
                "split": "test",
            }
            row.update(confusion_dict(y_true, pred))
            perf_rows.append(row)
            reference_rows.append(reference_check_row(config.key, model_label, row))

        base_correct = base_pred == y_true
        improved_correct = improved_pred == y_true
        table = [
            [int(np.sum(base_correct & improved_correct)), int(np.sum(base_correct & ~improved_correct))],
            [int(np.sum(~base_correct & improved_correct)), int(np.sum(~base_correct & ~improved_correct))],
        ]
        result = mcnemar(table, exact=False, correction=True)
        paired_rows.append(
            {
                "dataset": config.key,
                "test": "mcnemar",
                "metric": "classification_correctness",
                "estimate": float(result.statistic),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_value": float(result.pvalue),
                "note": "paired_predictions",
            }
        )
        for metric in ("accuracy", "recall", "fnr", "fpr", "f1"):
            observed, low, high = bootstrap_ci(
                y_true, base_pred, improved_pred, metric, args.bootstrap_iterations
            )
            paired_rows.append(
                {
                    "dataset": config.key,
                    "test": "bootstrap_ci",
                    "metric": metric,
                    "estimate": observed,
                    "ci_low": low,
                    "ci_high": high,
                    "p_value": np.nan,
                    "note": "improved_minus_base",
                }
            )

    perf = pd.DataFrame(perf_rows)
    reference_check = pd.DataFrame(reference_rows)
    if bool(reference_check["matches_reference"].all()):
        stats = pd.DataFrame(paired_rows)
        stats["reference_status"] = "matched_manuscript_confusion_matrices"
    else:
        stats = aggregate_reference_tests(select_datasets(args.dataset))
    perf.to_csv(REVISION_OUTPUTS_DIR / "performance_summary.csv", index=False)
    reference_check.to_csv(REVISION_OUTPUTS_DIR / "reference_confusion_check.csv", index=False)
    stats.to_csv(REVISION_OUTPUTS_DIR / "statistical_tests.csv", index=False)
    (REVISION_OUTPUTS_DIR / "statistical_tests.tex").write_text(
        stats.to_latex(index=False, float_format="%.4f"), encoding="utf-8"
    )
    print(perf.to_string(index=False))
    print(reference_check.to_string(index=False))
    print(stats.to_string(index=False))


def reference_check_row(dataset: str, model_label: str, observed: dict[str, object]) -> dict[str, object]:
    expected = REFERENCE_CONFUSIONS.get(dataset, {}).get(model_label)
    row = {
        "dataset": dataset,
        "model_label": model_label,
        "reference_available": expected is not None,
    }
    if expected is None:
        row.update({"matches_reference": False, "note": "no_reference_confusion_matrix"})
        return row
    for key in ("tp", "tn", "fp", "fn"):
        row[f"observed_{key}"] = int(observed[key])
        row[f"reference_{key}"] = int(expected[key])
        row[f"delta_{key}"] = int(observed[key]) - int(expected[key])
    row["matches_reference"] = all(row[f"delta_{key}"] == 0 for key in ("tp", "tn", "fp", "fn"))
    row["note"] = "matched" if row["matches_reference"] else "mismatch_use_aggregate_reference_tests"
    return row


def aggregate_reference_tests(configs: list[DatasetConfig]) -> pd.DataFrame:
    from statsmodels.stats.proportion import proportions_ztest

    rows = []
    for config in configs:
        refs = REFERENCE_CONFUSIONS.get(config.key)
        if not refs:
            continue
        base = refs["base"]
        improved = refs["improved"]
        specs = {
            "accuracy": (
                improved["tp"] + improved["tn"],
                sum(improved.values()),
                base["tp"] + base["tn"],
                sum(base.values()),
            ),
            "recall": (
                improved["tp"],
                improved["tp"] + improved["fn"],
                base["tp"],
                base["tp"] + base["fn"],
            ),
            "fnr": (
                improved["fn"],
                improved["tp"] + improved["fn"],
                base["fn"],
                base["tp"] + base["fn"],
            ),
            "fpr": (
                improved["fp"],
                improved["fp"] + improved["tn"],
                base["fp"],
                base["fp"] + base["tn"],
            ),
        }
        for metric, (imp_success, imp_n, base_success, base_n) in specs.items():
            imp_rate = imp_success / imp_n if imp_n else 0.0
            base_rate = base_success / base_n if base_n else 0.0
            statistic, p_value = proportions_ztest(
                count=np.array([imp_success, base_success]),
                nobs=np.array([imp_n, base_n]),
            )
            low, high = normal_diff_ci(imp_success, imp_n, base_success, base_n)
            rows.append(
                {
                    "dataset": config.key,
                    "test": "two_proportions_ztest",
                    "metric": metric,
                    "estimate": imp_rate - base_rate,
                    "ci_low": low,
                    "ci_high": high,
                    "p_value": float(p_value),
                    "reference_status": "mismatch_used_manuscript_aggregate_counts",
                    "note": "improved_minus_base",
                }
            )
        base_f1 = f1_from_counts(base)
        improved_f1 = f1_from_counts(improved)
        rows.append(
            {
                "dataset": config.key,
                "test": "aggregate_difference",
                "metric": "f1",
                "estimate": improved_f1 - base_f1,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_value": np.nan,
                "reference_status": "mismatch_used_manuscript_aggregate_counts",
                "note": "f1_has_no_simple_two_proportion_test",
            }
        )
    return pd.DataFrame(rows)


def normal_diff_ci(
    success_a: int, n_a: int, success_b: int, n_b: int, z_value: float = 1.96
) -> tuple[float, float]:
    rate_a = success_a / n_a if n_a else 0.0
    rate_b = success_b / n_b if n_b else 0.0
    diff = rate_a - rate_b
    se = np.sqrt((rate_a * (1 - rate_a) / n_a) + (rate_b * (1 - rate_b) / n_b))
    return float(diff - z_value * se), float(diff + z_value * se)


def f1_from_counts(counts: dict[str, int]) -> float:
    denominator = (2 * counts["tp"]) + counts["fp"] + counts["fn"]
    return (2 * counts["tp"]) / denominator if denominator else 0.0


def run_aequitas(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from aequitas.group import Group
    except Exception as exc:
        Group = None
        print(f"WARNING: aequitas import failed; using local region metrics fallback: {exc}")

    rows = []
    for config in select_datasets(args.dataset):
        for model_label in select_model_labels(args.model):
            path = prediction_path(config, model_label, args.age_distribution)
            if not path.exists():
                raise FileNotFoundError(f"Missing prediction CSV: {path}")
            df = pd.read_csv(path)
            audit = df.drop(columns=[config.label_name], errors="ignore")
            audit = audit.rename(columns={"y_true": "label_value"}).copy()
            audit["score"] = audit["score"].astype(int)
            if Group is not None:
                g = Group()
                xtab, _ = g.get_crosstabs(audit, attr_cols=["region"])
            else:
                xtab = local_region_crosstab(audit)
            xtab["dataset"] = config.key
            xtab["model_label"] = model_label
            rows.append(xtab)

            metrics = xtab[["attribute_value", "tpr", "fnr", "for"]].copy()
            metrics = metrics.rename(columns={"attribute_value": "region"})
            output_dir = config.directory / "outputs"
            ensure_dir(output_dir)
            metrics.to_csv(
                output_dir / f"{config.short}_metrics_region_{model_label}_{args.age_distribution}.csv",
                index=False,
            )
            ax = metrics.set_index("region")[["tpr", "fnr", "for"]].plot(kind="bar", figsize=(12, 6))
            ax.set_ylabel("Rate")
            ax.set_title(f"{config.key} {model_label} region metrics")
            plt.tight_layout()
            plt.savefig(
                output_dir / f"{config.short}_aequitas_region_{model_label}_{args.age_distribution}.pdf",
                bbox_inches="tight",
                format="pdf",
            )
            plt.close("all")

    if rows:
        ensure_dir(REVISION_OUTPUTS_DIR)
        pd.concat(rows, ignore_index=True).to_csv(
            REVISION_OUTPUTS_DIR / "aequitas_region_crosstabs.csv", index=False
        )


def local_region_crosstab(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region, group in audit.groupby("region", dropna=False):
        y_true = group["label_value"].to_numpy(dtype=int)
        y_pred = group["score"].to_numpy(dtype=int)
        metrics = confusion_dict(y_true, y_pred)
        rows.append(
            {
                "attribute_name": "region",
                "attribute_value": region,
                "tpr": metrics["recall"],
                "fnr": metrics["fnr"],
                "for": metrics["for"],
                "fpr": metrics["fpr"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
                "group_size": len(group),
            }
        )
    return pd.DataFrame(rows)


def run_all(args: argparse.Namespace) -> None:
    if args.model != "all":
        raise ValueError("run-all requires --model all because run-stats needs paired base/improved CSVs")
    run_models(args)
    run_aequitas(args)
    run_stats(args)


def select_datasets(dataset: str) -> list[DatasetConfig]:
    if dataset == "all":
        return list(DATASETS.values())
    return [DATASETS[dataset]]


def select_model_labels(model: str) -> list[str]:
    if model == "all":
        return list(MODEL_LABELS)
    return [model]


def add_common_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASETS.keys()],
        default="all",
        help="Dataset to process.",
    )
    parser.add_argument(
        "--age-distribution",
        choices=AGE_DISTRIBUTIONS,
        default="normal",
        help="Synthetic age distribution.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible experiment outputs for the Scientific Reports revision."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_models_parser = subparsers.add_parser("run-models", help="Train models and write predictions.")
    add_common_filters(run_models_parser)
    run_models_parser.add_argument(
        "--model", choices=["all", *MODEL_LABELS], default="all", help="Model variant to run."
    )
    run_models_parser.add_argument(
        "--skip-shap", action="store_true", help="Skip SHAP summaries and PDFs."
    )
    run_models_parser.add_argument(
        "--shap-sample",
        type=int,
        default=1000,
        help="Rows used for SHAP. Use -1 for all rows or 0 to disable.",
    )
    run_models_parser.set_defaults(func=run_models)

    aequitas_parser = subparsers.add_parser("run-aequitas", help="Write Aequitas region metrics.")
    add_common_filters(aequitas_parser)
    aequitas_parser.add_argument(
        "--model", choices=["all", *MODEL_LABELS], default="all", help="Model variant to audit."
    )
    aequitas_parser.set_defaults(func=run_aequitas)

    stats_parser = subparsers.add_parser("run-stats", help="Run paired statistical tests.")
    add_common_filters(stats_parser)
    stats_parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=2000,
        help="Bootstrap samples for confidence intervals.",
    )
    stats_parser.set_defaults(func=run_stats)

    run_all_parser = subparsers.add_parser("run-all", help="Run models, Aequitas, and stats.")
    add_common_filters(run_all_parser)
    run_all_parser.add_argument(
        "--model", choices=["all", *MODEL_LABELS], default="all", help="Model variant to run."
    )
    run_all_parser.add_argument(
        "--skip-shap", action="store_true", help="Skip SHAP summaries and PDFs."
    )
    run_all_parser.add_argument(
        "--shap-sample", type=int, default=1000, help="Rows used for SHAP."
    )
    run_all_parser.add_argument(
        "--bootstrap-iterations", type=int, default=2000, help="Bootstrap samples."
    )
    run_all_parser.set_defaults(func=run_all)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
