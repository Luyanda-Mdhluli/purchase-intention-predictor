# Purchase Intention Predictor

A Streamlit app that predicts a subsistence-retail customer's purchase intention from
five survey inputs, using a pre-trained Random Forest model. The result reads as a
hero confidence figure plus one bar per outcome, with only the predicted class
carrying colour.

## Inputs

| Input | Model feature | Scale |
| --- | --- | --- |
| Perceived Product Quality | `PPQ1` | 1-5 |
| Ability to Save Money at Local Store | `PV3` | 1-5 |
| Affordability of Product Offering | `PV2` | 1-5 |
| Age Group | `Age` | 1-5 |
| Gender | `Gender` | 1-3 |

The model outputs three classes: `0` no intention, `1` strong intention, `2` neutral
intention.

## Requirements

Python 3.11 or newer (tested on 3.11 and 3.14). The app warns at startup if the
installed scikit-learn is not the version `random_forest_model.pkl` was pickled with.

## Running it locally

```
python -m venv .venv
.venv/Scripts/activate        # Windows (PowerShell or Git Bash)
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

It opens on http://localhost:8501. Set a profile, then click **Predict intention**.
Results persist while you change the controls; a note appears when they are stale.

`random_forest_model.pkl` must sit beside `app.py`.

## Interface

The theme lives in `.streamlit/config.toml`; `app.py` layers a frosted-glass treatment
over it. Two notes for anyone editing that CSS:

- The cards are targeted by the `st-key-*` class Streamlit emits for a **keyed**
  container (`st.container(border=True, key=...)`). The generic layout-wrapper testid
  is shared with plain column wrappers, so styling that instead also glasses the
  nested Age/Gender columns.
- Outcome bars are plain HTML/CSS rather than a chart library, which is why there is
  no charting dependency.

## Deployment

Ready for GitHub Codespaces — the devcontainer runs Python 3.11, installs
`requirements.txt` and starts the app on port 8501 automatically.

## Retraining

`train.py` rebuilds the model from `Subsistence Retail Consumer Data.xlsx`:

```
python train.py                 # overwrites random_forest_model.pkl
python train.py -o /tmp/rf.pkl  # writes elsewhere
```

It reconstructs the pipeline from the original Colab notebook
("Assignment 03 Luyanda Mdhluli.ipynb"), with two deliberate differences, both
documented in the script: it trains on the five features the app actually collects
rather than all 37 columns, and it fixes `random_state` so runs are reproducible. The
original notebook set no `random_state`, so `random_forest_model.pkl` as committed
cannot be reproduced exactly by anything.

### How the target is derived

The `PI1` survey item (1-5) is collapsed to three classes: 1 and 2 become `0`
(no intention), 4 and 5 become `1` (strong intention), and 3 becomes `2` (neutral).
`PI2`-`PI4` are dropped as they measure the same construct as the target.

### Accuracy, honestly

The ~95% accuracy quoted in the original notebook's write-up belongs to the
**37-feature** model. The 5-feature model this app ships scores **0.86** on the
hold-out split, and recall on class `0` (no intention) is only 0.50 — there are just
28 such rows out of 281 (`{0: 28, 1: 196, 2: 57}`). Treat "no intention" predictions
with caution.
