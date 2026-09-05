# Purchase Intention Predictor

A Streamlit app that predicts a subsistence-retail customer's purchase intention from
five survey inputs, using a pre-trained Random Forest model, and shows the predicted
class probabilities as a donut chart.

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

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/purchase-intention-predictor.git
   cd purchase-intention-predictor
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Ensure `random_forest_model.pkl` is in the same directory as `app.py`.

## Usage

```
streamlit run app.py
```

Adjust the sliders and dropdowns, then click **Make Prediction**. Results persist while
you change inputs; a note appears when they are stale.

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
