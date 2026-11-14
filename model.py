"""Train a baseline and a Random Forest, evaluate on held-out data, check fairness slices.

Run:  python model.py              (train, evaluate, save charts)
      python model.py --demo       (same, then start the interactive risk-tier demo)
"""

import argparse

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from data_loader import RANDOM_STATE, load_data
from utils import (
    DISCLAIMER,
    interactive_loop,
    save_confusion_matrix,
    save_feature_importance,
    save_group_metrics,
)


def build_baseline() -> Pipeline:
    """Always predicts the most common class. Any real model must beat this."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", DummyClassifier(strategy="most_frequent")),
    ])


def build_random_forest() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=3,
            class_weight="balanced", random_state=RANDOM_STATE,
        )),
    ])


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        # zero_division=0: the baseline may never predict "disease", so precision is undefined
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def print_report(name, y_true, y_pred):
    m = metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n=== {name} (held-out test set, n={len(y_true)}) ===")
    print(f"Accuracy : {m['accuracy']:.3f}")
    print(f"Precision: {m['precision']:.3f}   (of patients flagged sick, how many were)")
    print(f"Recall   : {m['recall']:.3f}   (of sick patients, how many we caught)")
    print("Confusion matrix (rows = actual, cols = predicted; 0 = no disease, 1 = disease):")
    print(pd.DataFrame(cm, index=["actual 0", "actual 1"], columns=["pred 0", "pred 1"]).to_string())
    return m


def fairness_slices(model, X_test, y_test) -> pd.DataFrame:
    """Per-group test metrics for sex and age band, to spot performance gaps."""
    y_pred = pd.Series(model.predict(X_test), index=X_test.index)
    groups = {
        "Sex": X_test["sex"].map({1.0: "Male", 0.0: "Female"}),
        "Age": pd.cut(X_test["age"], bins=[0, 54, 200], labels=["Under 55", "55+"]),
    }
    rows = []
    for feature, labels in groups.items():
        for group in labels.dropna().unique():
            idx = labels[labels == group].index
            m = metrics(y_test.loc[idx], y_pred.loc[idx])
            rows.append({"feature": feature, "group": str(group), "n": len(idx),
                         "positives": int(y_test.loc[idx].sum()), **m})
    return pd.DataFrame(rows).sort_values(["feature", "group"]).reset_index(drop=True)


def main(run_demo: bool = False):
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {len(X_train)} patients   Test: {len(X_test)} patients (held out)")

    baseline = build_baseline().fit(X_train, y_train)
    forest = build_random_forest()

    # 5-fold cross-validation on the TRAINING set only: a steadier estimate than one small test set.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_recall = cross_val_score(forest, X_train, y_train, cv=cv, scoring="recall")
    cv_acc = cross_val_score(forest, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"\nRandom Forest 5-fold CV (train only): accuracy {cv_acc.mean():.3f} ± {cv_acc.std():.3f}, "
          f"recall {cv_recall.mean():.3f} ± {cv_recall.std():.3f}")

    forest.fit(X_train, y_train)

    base_pred = baseline.predict(X_test)
    rf_pred = forest.predict(X_test)
    base_m = print_report("Baseline: DummyClassifier (most frequent)", y_test, base_pred)
    rf_m = print_report("Random Forest", y_test, rf_pred)

    print("\n=== Baseline vs Random Forest ===")
    print(pd.DataFrame({"Baseline": base_m, "Random Forest": rf_m}).round(3).to_string())

    print("\n=== Fairness check: Random Forest by demographic group ===")
    slices = fairness_slices(forest, X_test, y_test)
    print(slices.round(3).to_string(index=False))
    for feature, part in slices.groupby("feature"):
        gap = part["recall"].max() - part["recall"].min()
        flag = "  <-- gap worth discussing" if gap >= 0.10 else ""
        print(f"Recall gap across {feature}: {gap:.3f}{flag}")
    small = slices[slices["n"] < 30]
    if not small.empty:
        print(f"Note: {', '.join(small['group'])} have fewer than 30 test patients, "
              "so their numbers are noisy. Mention this in the write-up.")

    print("\nSaving charts...")
    for path in (
        save_confusion_matrix(y_test, base_pred, "Baseline (most frequent)", "confusion_matrix_baseline.png"),
        save_confusion_matrix(y_test, rf_pred, "Random Forest", "confusion_matrix_random_forest.png"),
        save_feature_importance(forest, list(X_train.columns)),
        save_group_metrics(slices),
    ):
        print(f"  {path}")

    print(f"\n{DISCLAIMER}")

    if run_demo:
        interactive_loop(forest, X_train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Veritas AI heart-disease risk stratification MVP")
    parser.add_argument("--demo", action="store_true", help="start the interactive risk-tier demo after training")
    main(run_demo=parser.parse_args().demo)
