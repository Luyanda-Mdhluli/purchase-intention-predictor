"""Reproduce random_forest_model.pkl from Subsistence Retail Consumer Data.xlsx.

Reconstructed from the Colab notebook "Assignment 03 Luyanda Mdhluli.ipynb".

Two things differ from that notebook, both deliberate:

1. The notebook trains on every column except the target (37 features). The shipped
   random_forest_model.pkl was trained on only the five features the Streamlit app
   collects, so that is what this script does. The five were chosen in the Bayesian
   Network notebook by ranking cross-category correlations. Note that the ~95%
   accuracy quoted in the notebook's write-up belongs to the 37-feature model; the
   5-feature model scores lower. Set FEATURES = ALL_FEATURES to get the former.

2. The notebook used a bare RandomForestClassifier() with no random_state, so the
   original pickle cannot be reproduced bit-for-bit by anything. RANDOM_STATE is
   fixed here so that this script is reproducible going forward.

Usage:
    python train.py                 # writes random_forest_model.pkl (overwrites!)
    python train.py -o /tmp/rf.pkl  # writes elsewhere
"""

import argparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_FILE = "Subsistence Retail Consumer Data.xlsx"
DEFAULT_OUTPUT = "random_forest_model.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET = "PI1"
# The other purchase-intention items are dropped: they measure the same construct as
# the target and would leak.
DROPPED = ["PI2", "PI3", "PI4"]

# The survey's 1-5 Likert target collapsed to three classes. 1 and 2 are both "no
# intention" and 4 and 5 are both "intention", so they are merged; 3 (neutral) becomes
# its own class. This matches CLASS_LABELS in app.py.
TARGET_MAP = {1: 0, 2: 0, 3: 2, 4: 1, 5: 1}

FEATURES = ["PPQ1", "PV3", "PV2", "Age", "Gender"]


def load_dataset(path=DATA_FILE):
    """Return (X, y) with the target collapsed to three classes."""
    df = pd.read_excel(path).drop(columns=DROPPED)

    unmapped = set(df[TARGET].unique()) - set(TARGET_MAP)
    if unmapped:
        raise ValueError(f"Unexpected {TARGET} values, cannot map to classes: {sorted(unmapped)}")
    y = df[TARGET].map(TARGET_MAP)

    all_features = [c for c in df.columns if c != TARGET]
    missing = set(FEATURES) - set(all_features)
    if missing:
        raise ValueError(f"Features missing from {path}: {sorted(missing)}")

    return df[FEATURES], y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--data", default=DATA_FILE, help="path to the survey spreadsheet")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="where to write the model")
    args = parser.parse_args()

    X, y = load_dataset(args.data)
    print(f"{len(X)} rows, {len(FEATURES)} features: {', '.join(FEATURES)}")
    print(f"class balance: {y.value_counts().sort_index().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    print("\nHold-out performance (0=no intention, 1=strong, 2=neutral):")
    print(classification_report(y_test, model.predict(X_test), zero_division=0))
    print("confusion matrix:")
    print(confusion_matrix(y_test, model.predict(X_test)))
    print("\nfeature importances:")
    for name, importance in zip(FEATURES, model.feature_importances_):
        print(f"  {name:<8} {importance:.3f}")

    joblib.dump(model, args.output)
    print(f"\nsaved -> {args.output}")


if __name__ == "__main__":
    main()
