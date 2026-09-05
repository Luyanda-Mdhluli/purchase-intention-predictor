"""Streamlit app predicting subsistence-retail purchase intention from survey inputs.

The model is a RandomForestClassifier trained on `Subsistence Retail Consumer Data.xlsx`
and shipped as `random_forest_model.pkl`. See train.py to regenerate it.
"""

import os

import joblib
import pandas as pd
import sklearn
import streamlit as st

# The pickle was created with this scikit-learn version. Loading it under a different
# version is not guaranteed to reproduce the same predictions, so we surface a warning
# rather than let it fail quietly. requirements.txt pins the matching version; rerun
# train.py after changing it.
MODEL_SKLEARN_VERSION = "1.9.0"
MODEL_FILENAME = "random_forest_model.pkl"

# Column order the model was fitted on. Passing a DataFrame with these names (rather
# than a bare array) keeps the app honest if the model is ever retrained on a
# different ordering, and silences sklearn's missing-feature-names warning.
FEATURE_ORDER = ["PPQ1", "PV3", "PV2", "Age", "Gender"]

CLASS_LABELS = {0: "No intention", 1: "Strong intention", 2: "Neutral"}

# The classes are an ordered scale, so read them in that order rather than by class
# value: no intention -> neutral -> strong intention.
CLASS_DISPLAY_ORDER = [0, 2, 1]

# Hue follows the class, never its rank, so a class keeps its colour between runs.
# These are the first three categorical slots, validated for contrast and
# colour-vision separation against this app's glass surface.
CLASS_COLORS = {0: "#d95926", 2: "#3987e5", 1: "#199e70"}

CLASS_BLURB = {
    0: "Unlikely to buy from the local store.",
    1: "Likely to buy from the local store.",
    2: "Undecided, could go either way.",
}

QUALITY_LABELS = {1: "Very poor", 2: "Poor", 3: "Moderate", 4: "Good", 5: "Very good"}
SAVING_LABELS = {
    1: "Cannot save",
    2: "Rarely saves",
    3: "Sometimes saves",
    4: "Can save",
    5: "Always saves",
}
AFFORD_LABELS = {
    1: "Unaffordable",
    2: "Barely affordable",
    3: "Fairly affordable",
    4: "Affordable",
    5: "Very affordable",
}
AGE_LABELS = {1: "18-22", 2: "23-28", 3: "29-35", 4: "36-49", 5: "50-65"}
AGE_NOTE = {
    1: "Gen Z",
    2: "Millennials",
    3: "Young professionals",
    4: "Prime buyers",
    5: "Boomers",
}
GENDER_LABELS = {1: "Male", 2: "Female", 3: "Prefer not to say"}

st.set_page_config(
    page_title="Purchase Intention Predictor", page_icon="◑", layout="wide"
)

# Base palette comes from .streamlit/config.toml; everything below is the glass layer.
# Streamlit's own surfaces are flattened to transparent so the gradient shows through
# and the frosted panels have something to refract.
GLASS_CSS = """
<style>
:root {
    --glass: rgba(255, 255, 255, 0.06);
    --glass-edge: rgba(255, 255, 255, 0.14);
    --glass-hi: rgba(255, 255, 255, 0.28);
    --ink: #ecebf5;
    --ink-dim: #a9a7c4;
    --ink-faint: #7f7d9c;
}

/* The gradient the glass sits on. Fixed, so scrolling does not drag it. */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(900px 700px at 12% 8%,  rgba(67, 56, 202, 0.55), transparent 60%),
        radial-gradient(800px 600px at 88% 4%,  rgba(124, 58, 237, 0.45), transparent 60%),
        radial-gradient(900px 800px at 78% 92%, rgba(15, 118, 110, 0.42), transparent 62%),
        #07070f;
    background-attachment: fixed;
}
[data-testid="stHeader"], [data-testid="stAppHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stAppDeployButton"] { display: none; }
[data-testid="stMainBlockContainer"], .block-container {
    padding-top: 2.6rem;
    padding-bottom: 3rem;
    max-width: 1180px;
}

/* The two cards become frosted panels. Targeted by the st-key-* class that
   Streamlit emits for a keyed container: the generic layout-wrapper testid is
   shared with plain column wrappers, so styling that would glass the nested
   Age/Gender columns too. */
.st-key-profile_card,
.st-key-prediction_card {
    background: var(--glass) !important;
    backdrop-filter: blur(28px) saturate(165%);
    -webkit-backdrop-filter: blur(28px) saturate(165%);
    border: 1px solid var(--glass-edge) !important;
    border-radius: 22px !important;
    box-shadow:
        0 18px 50px rgba(0, 0, 0, 0.38),
        inset 0 1px 0 var(--glass-hi);
    padding: 1.55rem 1.7rem 1.7rem !important;
}

/* Header */
.pi-title {
    font-size: 2.35rem;
    font-weight: 600;
    letter-spacing: -0.025em;
    color: var(--ink);
    margin: 0 0 0.3rem;
    line-height: 1.1;
}
.pi-sub {
    color: var(--ink-dim);
    font-size: 0.97rem;
    margin: 0 0 1.8rem;
    max-width: 62ch;
}
.pi-card-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--ink-faint);
    margin: 0 0 1.25rem;
}

/* Widgets: strip Streamlit's opaque chrome so the glass reads through. */
[data-testid="stWidgetLabel"] p {
    font-size: 0.9rem !important;
    font-weight: 500;
    color: var(--ink) !important;
}
[data-baseweb="select"] > div {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid var(--glass-edge) !important;
    border-radius: 12px !important;
}
[data-testid="stSelectbox"] svg { fill: var(--ink-dim); }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {
    color: var(--ink-faint) !important;
    background: transparent !important;
}

.stButton > button {
    width: 100%;
    backdrop-filter: blur(14px) saturate(150%);
    -webkit-backdrop-filter: blur(14px) saturate(150%);
    background: linear-gradient(180deg, rgba(255,255,255,0.20), rgba(255,255,255,0.07));
    color: var(--ink);
    font-weight: 600;
    letter-spacing: 0.01em;
    border: 1px solid var(--glass-edge);
    border-radius: 14px;
    padding: 0.68rem 1rem;
    box-shadow: inset 0 1px 0 var(--glass-hi), 0 8px 22px rgba(0,0,0,0.28);
    transition: transform 0.15s ease, background 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(180deg, rgba(255,255,255,0.28), rgba(255,255,255,0.12));
    transform: translateY(-1px);
    color: #ffffff;
}
.stButton > button:active { transform: translateY(0); }

[data-testid="stAlert"] {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid var(--glass-edge);
    border-radius: 14px;
    backdrop-filter: blur(12px);
}

/* Result: eyebrow, the answer, then the hero figure. */
.pi-eyebrow {
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--ink-faint);
    margin-bottom: 0.5rem;
}
.pi-answer {
    font-size: 1.6rem;
    font-weight: 600;
    letter-spacing: -0.015em;
    line-height: 1.15;
    margin-bottom: 0.15rem;
}
.pi-blurb { color: var(--ink-dim); font-size: 0.9rem; margin-bottom: 1.1rem; }
.pi-hero {
    font-size: 3.6rem;
    font-weight: 600;
    letter-spacing: -0.03em;
    line-height: 1;
    color: var(--ink);
}
.pi-hero-cap {
    font-size: 0.78rem;
    color: var(--ink-faint);
    margin-top: 0.25rem;
    margin-bottom: 1.6rem;
}

/* Emphasis bars: the predicted class carries its hue, the rest recede. */
.pi-bar {
    display: grid;
    grid-template-columns: 8.6rem 1fr 3.4rem;
    align-items: center;
    gap: 0.85rem;
    margin-bottom: 0.7rem;
}
.pi-bar-name { font-size: 0.87rem; color: var(--ink-dim); }
.pi-bar-name.is-lead { color: var(--ink); font-weight: 600; }
.pi-track {
    height: 9px;
    background: rgba(255, 255, 255, 0.07);
    border-radius: 0 4px 4px 0;
    overflow: hidden;
}
.pi-fill { height: 100%; border-radius: 0 4px 4px 0; }
.pi-bar-val {
    font-size: 0.87rem;
    text-align: right;
    color: var(--ink-dim);
    font-variant-numeric: tabular-nums;
}
.pi-bar-val.is-lead { color: var(--ink); font-weight: 600; }

.pi-stale { font-size: 0.8rem; color: #e0b341; margin-top: 1rem; }
.pi-placeholder {
    color: var(--ink-faint);
    font-size: 0.92rem;
    padding: 2.6rem 0 2.8rem;
    text-align: center;
}
</style>
"""

st.markdown(GLASS_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found. Ensure '{MODEL_FILENAME}' sits beside this script.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def predict(model, inputs):
    """Return (predicted_class, {class_value: probability}) for one profile."""
    features = pd.DataFrame([inputs], columns=FEATURE_ORDER)
    predicted_class = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    return predicted_class, dict(zip(model.classes_, probabilities))


def result_html(predicted_class, probabilities):
    """Hero figure plus one emphasis bar per class, ordered along the scale."""
    accent = CLASS_COLORS.get(predicted_class, "#8b7cf6")
    confidence = probabilities.get(predicted_class, 0.0) * 100

    rows = []
    for value in CLASS_DISPLAY_ORDER:
        if value not in probabilities:
            continue
        pct = probabilities[value] * 100
        lead = value == predicted_class
        # Only the predicted class carries colour; the others stay recessive so the
        # answer is not buried in three competing hues.
        fill = CLASS_COLORS[value] if lead else "rgba(255,255,255,0.22)"
        cls = " is-lead" if lead else ""
        rows.append(
            '<div class="pi-bar" title="{name}: {pct:.1f}%">'
            '<div class="pi-bar-name{cls}">{name}</div>'
            '<div class="pi-track"><div class="pi-fill" '
            'style="width:{pct:.1f}%;background:{fill}"></div></div>'
            '<div class="pi-bar-val{cls}">{pct:.1f}%</div>'
            "</div>".format(name=CLASS_LABELS[value], pct=pct, fill=fill, cls=cls)
        )

    return (
        '<div class="pi-eyebrow">Predicted outcome</div>'
        '<div class="pi-answer" style="color:{accent}">{label}</div>'
        '<div class="pi-blurb">{blurb}</div>'
        '<div class="pi-hero">{conf:.1f}%</div>'
        '<div class="pi-hero-cap">confidence in this prediction</div>'
        "{rows}".format(
            accent=accent,
            label=CLASS_LABELS[predicted_class],
            blurb=CLASS_BLURB.get(predicted_class, ""),
            conf=confidence,
            rows="".join(rows),
        )
    )


st.markdown('<div class="pi-title">Purchase Intention Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="pi-sub">A random forest over subsistence-retail survey responses. '
    "Set a customer profile and the model estimates how likely that shopper is to buy "
    "from their local store.</div>",
    unsafe_allow_html=True,
)

model = load_model()
if model is not None and sklearn.__version__ != MODEL_SKLEARN_VERSION:
    st.warning(
        f"The model was trained with scikit-learn {MODEL_SKLEARN_VERSION} but "
        f"{sklearn.__version__} is installed. Predictions may be unreliable. Install "
        "the pinned requirements, or rerun train.py."
    )

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    with st.container(border=True, key="profile_card"):
        st.markdown('<div class="pi-card-title">Customer profile</div>', unsafe_allow_html=True)

        ppq1 = st.select_slider(
            "Perceived product quality",
            options=list(QUALITY_LABELS),
            value=3,
            format_func=QUALITY_LABELS.get,
            key="ppq1",
        )
        pv3 = st.select_slider(
            "Ability to save money at the local store",
            options=list(SAVING_LABELS),
            value=3,
            format_func=SAVING_LABELS.get,
            key="pv3",
        )
        pv2 = st.select_slider(
            "Affordability of the product offering",
            options=list(AFFORD_LABELS),
            value=3,
            format_func=AFFORD_LABELS.get,
            key="pv2",
        )

        sub1, sub2 = st.columns(2)
        with sub1:
            age = st.selectbox(
                "Age group",
                list(AGE_LABELS),
                format_func=lambda v: "{}  ({})".format(AGE_LABELS[v], AGE_NOTE[v]),
            )
        with sub2:
            gender = st.selectbox("Gender", list(GENDER_LABELS), format_func=GENDER_LABELS.get)

        st.write("")
        predict_button = st.button("Predict intention", width="stretch")

current_inputs = [ppq1, pv3, pv2, age, gender]

# Results are kept in session state so that nudging a control after predicting does
# not wipe the panel; we just flag that it is now stale.
if predict_button and model is not None:
    try:
        predicted_class, probabilities = predict(model, current_inputs)
        st.session_state["result"] = {
            "inputs": current_inputs,
            "predicted_class": predicted_class,
            "probabilities": probabilities,
        }
        st.session_state.pop("error", None)
    except Exception as e:
        st.session_state.pop("result", None)
        st.session_state["error"] = str(e)

with col2:
    with st.container(border=True, key="prediction_card"):
        st.markdown('<div class="pi-card-title">Prediction</div>', unsafe_allow_html=True)

        result = st.session_state.get("result")
        if model is None:
            st.markdown(
                '<div class="pi-placeholder">Model unavailable.</div>', unsafe_allow_html=True
            )
        elif "error" in st.session_state:
            st.error("Error during prediction: {}".format(st.session_state["error"]))
        elif result is None:
            st.markdown(
                '<div class="pi-placeholder">Set a profile, then hit '
                "<b>Predict intention</b>.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                result_html(result["predicted_class"], result["probabilities"]),
                unsafe_allow_html=True,
            )
            if result["inputs"] != current_inputs:
                st.markdown(
                    '<div class="pi-stale">Profile changed, predict again to refresh.</div>',
                    unsafe_allow_html=True,
                )
            with st.expander("View as table"):
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Outcome": [CLASS_LABELS[v] for v in CLASS_DISPLAY_ORDER],
                            "Probability": [
                                "{:.1f}%".format(result["probabilities"].get(v, 0.0) * 100)
                                for v in CLASS_DISPLAY_ORDER
                            ],
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
