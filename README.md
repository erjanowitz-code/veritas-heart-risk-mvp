# Veritas AI Heart-Disease Risk Stratification MVP

> **Prototype, not medical advice.** This is a student hackathon project. It was trained on
> 303 patients from one clinic in 1988 and has not been clinically validated. Do not use it
> to make health decisions; talk to a doctor.

**Event:** Veritas AI International High School Hackathon 2026 · **Track:** AI + Health ·
**Team:** _add team member names_

The model takes a patient's clinical measurements (age, blood pressure, cholesterol,
exercise stress test results and more) and gives back a **Low / Moderate / High** heart-disease
risk tier. It is compared against a do-nothing baseline and checked for performance gaps
across sex and age groups.

## Quick start

Requires Python 3.13.

```bash
pip install -r requirements.txt
python data_loader.py        # download the data, show cleaning and the train/test split
python model.py              # train, evaluate, run the fairness check, save charts to charts/
python model.py --demo       # same, then type in patient vitals to get a risk tier
```

In the demo, press Enter to use the typical (median) value for any field, or type `q` to quit.
The first run downloads the dataset from Hugging Face. On Windows you may see harmless warnings
about "symlinks" and "unauthenticated requests".

## Project structure

| File | What it does |
|---|---|
| `data_loader.py` | Loads the dataset with Hugging Face `datasets`, turns `?` placeholders into missing values, makes a stratified 80/20 train/test split |
| `model.py` | Builds scikit-learn Pipelines (median imputation + classifier), compares a `DummyClassifier` baseline with a `RandomForestClassifier`, runs 5-fold cross-validation, prints test metrics and per-group results |
| `utils.py` | Saves the charts to `charts/` and runs the interactive risk-tier demo |
| `charts/` | Confusion matrices, feature importance, performance by group |

## Data

[`skrishna/heart_disease_uci`](https://huggingface.co/datasets/skrishna/heart_disease_uci) on
Hugging Face: the Cleveland subset of the UCI Heart Disease dataset. It has 303 patients and 13
features, and 139 of the patients (46%) have heart disease.
- 6 values are missing (4 in `ca`, 2 in `thal`). They are filled with the **training-set**
  median inside the Pipeline, so no information from the test set leaks into training.
- The split is stratified, so the training set (242 patients) and the test set (61 patients)
  have the same share of patients with heart disease.

## Results

Held-out test set, 61 patients:

| Metric | Baseline (always "no disease") | Random Forest |
|---|---|---|
| Accuracy | 0.541 | 0.918 |
| Precision | 0.000 | 0.871 |
| Recall | 0.000 | **0.964** |

5-fold cross-validation on the training set: **accuracy 0.80 ± 0.02, recall 0.76 ± 0.08**.

The Random Forest missed 1 of 28 sick patients and wrongly flagged 4 healthy ones. We focus on
**recall** because missing a sick patient is worse than an extra check-up. The test score is
higher than the cross-validation score, and with only 61 test patients that gap is likely partly
luck, so **we treat ~0.80 accuracy as the more realistic estimate**.

The most important features are the thallium stress test, chest pain type and the number of
major vessels seen on fluoroscopy (`charts/feature_importance.png`).

### Fairness check

| Group | Test patients | Accuracy | Recall |
|---|---|---|---|
| Female | 20 | 1.000 | 1.000 |
| Male | 41 | 0.878 | 0.952 |
| Under 55 | 28 | 1.000 | 1.000 |
| 55+ | 33 | 0.848 | 0.952 |

All of the model's errors were on men and on patients aged 55 and over. The female and under-55
groups have fewer than 30 test patients each, which is **too few to conclude that the model is
fairer for them**. The perfect scores there are probably noise.

## Limitations

- **Small, old, narrow data:** 303 patients from a single US clinic in 1988, about two-thirds
  of them men. The results may not hold for other populations or for today's patients.
- **Hospital-only inputs:** the strongest features (thallium test, fluoroscopy) come from
  hospital tests, so this cannot yet be used as an at-home screening tool.
- **Risk tiers are a prototype choice:** Low is under 33% predicted probability, Moderate is
  33–66% and High is above 66%. These are not clinical cut-offs.
- **Small test set:** a single 61-patient test set makes every number noisy, which is why we also
  report cross-validation.

## Sources

- **Dataset:** Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). *Heart Disease*
  [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X — loaded from the
  Hugging Face mirror [`skrishna/heart_disease_uci`](https://huggingface.co/datasets/skrishna/heart_disease_uci).
- **Libraries:** pandas, NumPy, scikit-learn, matplotlib, seaborn, Hugging Face `datasets`.
- **AI tools:** Claude Code (Anthropic) helped write and debug the code. The team reviewed and
  can explain every part of it.
