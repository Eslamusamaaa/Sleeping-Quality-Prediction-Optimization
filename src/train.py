"""Train and evaluate the Random Forest sleep-quality model.

About 98% of Final_DataSet.csv is synthetic, so a random train/test split mostly measures how well the model
re-learns the data generator. To measure real-world fit the real rows are split off first: the model is trained
on all synthetic rows plus part of the real rows and tested on the unseen real rows (repeated with several
seeds). A model trained on synthetic rows only is evaluated the same way for comparison.
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import config
from src.preprocessing import build_features, load_final_dataset, split_real_synthetic

N_REPEATS = 5
REAL_TEST_SIZE = 0.3


def new_model():
    return RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=3,
        max_features="sqrt", random_state=config.SEED, n_jobs=-1,
    )


def fit(X_syn, y_syn, X_real=None, y_real=None):
    """Fit on synthetic rows, optionally adding real rows with a higher sample weight."""
    if X_real is None:
        return new_model().fit(X_syn, y_syn)
    X = pd.concat([X_syn, X_real])
    y = pd.concat([y_syn, y_real])
    w = np.r_[np.ones(len(X_syn)), np.full(len(X_real), config.REAL_SAMPLE_WEIGHT)]
    return new_model().fit(X, y, sample_weight=w)


def scores(y_true, y_pred):
    err = np.abs(np.asarray(y_pred) - np.asarray(y_true))
    return {"accuracy": accuracy_score(y_true, y_pred), "within_1": float((err <= 1).mean()), "mae": float(err.mean())}


def evaluate(real, synthetic):
    X_syn, y_syn = build_features(synthetic), synthetic[config.TARGET]
    X_real, y_real = build_features(real), real[config.TARGET]
    strat = y_real.replace({4: 5})  # only 4-5 real rows have score 4; merge for stratification only

    runs, pooled_true, pooled_pred = [], [], []
    for seed in range(N_REPEATS):
        tr, te = train_test_split(real.index, test_size=REAL_TEST_SIZE, random_state=seed, stratify=strat)
        mixed = fit(X_syn, y_syn, X_real.loc[tr], y_real.loc[tr])
        synth_only = fit(X_syn, y_syn)
        pred = mixed.predict(X_real.loc[te])
        runs.append({
            "mixed": scores(y_real.loc[te], pred),
            "synthetic_only": scores(y_real.loc[te], synth_only.predict(X_real.loc[te])),
        })
        pooled_true.extend(y_real.loc[te]), pooled_pred.extend(pred)

    mean = lambda key, m: float(np.mean([r[key][m] for r in runs]))
    summary = {key: {m: round(mean(key, m), 4) for m in ("accuracy", "within_1", "mae")}
               for key in ("mixed", "synthetic_only")}

    # the number the original notebook reported: a random split of the (mostly synthetic) data
    X_tr, X_te, y_tr, y_te = train_test_split(
        pd.concat([X_syn, X_real]), pd.concat([y_syn, y_real]),
        test_size=0.2, random_state=config.SEED, stratify=pd.concat([y_syn, y_real]))
    summary["random_split_all_data"] = {"accuracy": round(accuracy_score(y_te, new_model().fit(X_tr, y_tr).predict(X_te)), 4)}
    return summary, pooled_true, pooled_pred


def save_plots(model, pooled_true, pooled_pred):
    config.IMG_DIR.mkdir(parents=True, exist_ok=True)
    labels = sorted(set(pooled_true) | set(pooled_pred))
    cm = confusion_matrix(pooled_true, pooled_pred, labels=labels)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"), plt.ylabel("Actual")
    plt.title(f"Confusion matrix on unseen REAL rows ({N_REPEATS} repeated hold-outs)")
    plt.tight_layout(), plt.savefig(config.IMG_DIR / "confusion_matrix.png", dpi=130), plt.close()

    imp = pd.Series(model.feature_importances_, index=config.FEATURES).sort_values()
    plt.figure(figsize=(7, 4.5))
    imp.plot.barh(color="#6c63ff")
    plt.title("Feature importance (final model)")
    plt.tight_layout(), plt.savefig(config.IMG_DIR / "feature_importance.png", dpi=130), plt.close()


def main():
    df = load_final_dataset()
    real, synthetic = split_real_synthetic(df)
    print(f"Rows: {len(synthetic)} synthetic, {len(real)} unique real")

    summary, pooled_true, pooled_pred = evaluate(real, synthetic)
    print("\nUnseen real rows (mean of", N_REPEATS, "hold-outs):")
    print(json.dumps(summary, indent=2))
    print(classification_report(pooled_true, pooled_pred, zero_division=0))

    # final model: every synthetic row + every unique real row
    final = fit(build_features(synthetic), synthetic[config.TARGET], build_features(real), real[config.TARGET])
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final, config.MODEL_PATH, compress=3)

    stats = {c: {"min": float(df[c].min()), "max": float(df[c].max()), "mean": round(float(df[c].mean()), 2)}
             for c in config.NUMERIC_FEATURES}
    config.STATS_PATH.write_text(json.dumps(stats, indent=2))
    config.METRICS_PATH.write_text(json.dumps(summary, indent=2))
    save_plots(final, pooled_true, pooled_pred)
    print(f"Saved {config.MODEL_PATH.name} ({config.MODEL_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
