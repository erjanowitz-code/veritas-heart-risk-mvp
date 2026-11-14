"""Chart saving and the interactive risk-tier demo."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save files without needing a display window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from data_loader import FEATURE_INFO

CHARTS_DIR = Path(__file__).parent / "charts"

DISCLAIMER = (
    "DISCLAIMER: This is a student hackathon prototype, NOT medical advice. It was trained on "
    "303 patients from one 1988 US clinic and has not been clinically validated. "
    "Do not use it to make health decisions; talk to a doctor."
)

# Probability of heart disease -> risk tier. Thresholds are a prototype choice, not clinical cut-offs.
TIERS = [(0.33, "Low"), (0.66, "Moderate"), (1.01, "High")]


def _save(fig, filename: str) -> Path:
    CHARTS_DIR.mkdir(exist_ok=True)
    path = CHARTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_confusion_matrix(y_true, y_pred, title: str, filename: str) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No disease", "Disease"], yticklabels=["No disease", "Disease"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix: {title}")
    return _save(fig, filename)


def save_feature_importance(pipeline, feature_names, filename="feature_importance.png") -> Path:
    importances = pd.Series(pipeline.named_steps["clf"].feature_importances_, index=feature_names)
    importances = importances.sort_values()
    labels = [FEATURE_INFO.get(f, f).split(" (")[0] for f in importances.index]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(labels, importances.values, color="#3b6fb6")
    ax.set_xlabel("Importance (mean decrease in impurity)")
    ax.set_title("Random Forest feature importance")
    return _save(fig, filename)


def save_group_metrics(slices: pd.DataFrame, filename="fairness_by_group.png") -> Path:
    data = slices.melt(id_vars=["feature", "group", "n"], value_vars=["accuracy", "recall"],
                       var_name="metric", value_name="score")
    data["label"] = data["group"] + " (n=" + data["n"].astype(str) + ")"
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=data, x="label", y="score", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Score on held-out test set")
    ax.set_title("Random Forest performance by demographic group")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    return _save(fig, filename)


def risk_tier(probability: float) -> str:
    for upper, tier in TIERS:
        if probability < upper:
            return tier
    return "High"


def _ask(feature: str, default: float) -> float:
    while True:
        raw = input(f"  {FEATURE_INFO[feature]} [Enter = {default:g}]: ").strip()
        if raw == "":
            return default
        if raw.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        try:
            return float(raw)
        except ValueError:
            print("    Please type a number (or press Enter for the default).")


def interactive_loop(pipeline, X_train: pd.DataFrame) -> None:
    """Ask for one patient's vitals, print a Low / Moderate / High risk tier, repeat."""
    defaults = X_train.median(numeric_only=True)
    print("\n" + "=" * 70)
    print("Veritas AI heart-disease risk demo")
    print("Press Enter to use the typical (median) value. Type 'q' at any prompt to quit.")
    print(DISCLAIMER)
    print("=" * 70)
    try:
        while True:
            print("\nNew patient:")
            patient = {f: _ask(f, float(defaults[f])) for f in X_train.columns}
            row = pd.DataFrame([patient], columns=X_train.columns)
            prob = float(pipeline.predict_proba(row)[0, 1])
            print(f"\n  Estimated probability of heart disease: {prob:.0%}")
            print(f"  Risk tier: {risk_tier(prob).upper()}")
            print(f"  {DISCLAIMER}")
            if input("\nAnother patient? [Y/n]: ").strip().lower() in {"n", "no", "q"}:
                break
    except (KeyboardInterrupt, EOFError):
        pass
    print("\nDemo finished.")
