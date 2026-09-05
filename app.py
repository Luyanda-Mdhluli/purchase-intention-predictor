"""Streamlit app predicting subsistence-retail purchase intention from survey inputs.

The model is a RandomForestClassifier trained on `Subsistence Retail Consumer Data.xlsx`
and shipped as `random_forest_model.pkl`.
"""

import os

import joblib
import pandas as pd
import plotly.graph_objects as go
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

# Keyed by class value so the chart stays correct regardless of the order of
# model.classes_.
CLASS_LABELS = {0: "No intention", 1: "Strong intention", 2: "Neutral intention"}
CLASS_COLORS = {0: "#fff3cc", 1: "#ffd700", 2: "#b38600"}
CLASS_MESSAGES = {
    0: "😔 Prediction: **No intention** to purchase.",
    1: "✅ Prediction: **Strong intention** to purchase!",
    2: "🤔 Prediction: **Neutral intention** to purchase.",
}

PPQ1_LABELS = {1: "🤢 Very poor", 2: "😬 Poor", 3: "😐 Moderate", 4: "😄 Good", 5: "🤩 Very good"}
PV3_LABELS = {
    1: "💸 Cannot save", 2: "💸 Rarely save", 3: "🤔 Sometimes save",
    4: "💵 Can save", 5: "💰 Always save",
}
PV2_LABELS = {
    1: "🚫 Unaffordable", 2: "😓 Barely affordable", 3: "🤷 Fairly affordable",
    4: "💳 Affordable", 5: "🤑 Very affordable",
}
AGE_LABELS = {
    1: "18-22 (Gen Z)", 2: "23-28 (Millennials)", 3: "29-35 (Young Professionals)",
    4: "36-49 (Prime Buyers)", 5: "50-65 (Boomers)",
}
GENDER_LABELS = {1: "Male", 2: "Female", 3: "Prefer not to say"}

st.set_page_config(page_title="Purchase Intention Predictor", page_icon="🛒", layout="wide")

# Base colours come from .streamlit/config.toml; only what the theme API cannot
# express is set here.
st.markdown(
    """
    <style>
    h1, h2, h3, h4, h5 { color: #b38600; }
    .stPlotlyChart {
        border: 2px solid #ffd700;
        padding: 5px;
        border-radius: 10px;
        background-color: #fff3cc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💡 Purchase Intention Predictor")


@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(
            f"🚨 Model file not found. Please ensure '{MODEL_FILENAME}' is in the same "
            "directory as this script."
        )
        return None
    except Exception as e:
        st.error(f"🚨 Error loading model: {e}")
        return None


with st.spinner("Loading model..."):
    model = load_model()

if model is None:
    st.error("Failed to load the model. Please check the model file path.")
elif sklearn.__version__ != MODEL_SKLEARN_VERSION:
    st.warning(
        f"⚠️ The model was trained with scikit-learn {MODEL_SKLEARN_VERSION} but "
        f"{sklearn.__version__} is installed. Predictions may be unreliable — install "
        "the pinned requirements to be sure."
    )
else:
    st.success("🎉 Model loaded successfully!")


def predict(model, inputs):
    """Return (predicted_class, {class_value: probability}) for one profile."""
    features = pd.DataFrame([inputs], columns=FEATURE_ORDER)
    predicted_class = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    return predicted_class, dict(zip(model.classes_, probabilities))


def probability_chart(probabilities):
    classes = sorted(probabilities)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=[CLASS_LABELS.get(c, f"Class {c}") for c in classes],
                values=[probabilities[c] * 100 for c in classes],
                hole=0.4,
                marker=dict(
                    colors=[CLASS_COLORS.get(c, "#cccccc") for c in classes],
                    line=dict(color="#000000", width=1.5),
                ),
                textfont=dict(color="#000000", size=14),
                hovertemplate="<b>%{label}</b><br>Probability: %{value:.1f}%<extra></extra>",
                texttemplate="%{label}<br>%{value:.1f}%",
            )
        ]
    )
    fig.update_layout(
        title=dict(
            text="🔎 Probability Distribution of Purchase Intention",
            font=dict(color="#000000", size=16),
            y=0.95,
        ),
        plot_bgcolor="rgba(255,249,230,0)",
        paper_bgcolor="rgba(255,249,230,0)",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        annotations=[
            dict(
                text="Purchase<br>Intention",
                x=0.5,
                y=0.5,
                font=dict(size=14, color="#000000"),
                showarrow=False,
            )
        ],
    )
    return fig


col1, col2 = st.columns(2)

with col1:
    st.header("📋 Customize Your Customer Profile:")

    ppq1 = st.slider("Perceived Product Quality", 1, 5, 3, key="ppq1")
    st.write(f"**Selected:** {PPQ1_LABELS[ppq1]}")

    pv3 = st.slider("Ability to Save Money at Local Store", 1, 5, 3, key="pv3")
    st.write(f"**Selected:** {PV3_LABELS[pv3]}")

    pv2 = st.slider("Affordability of Product Offering", 1, 5, 3, key="pv2")
    st.write(f"**Selected:** {PV2_LABELS[pv2]}")

    age = st.selectbox("Select Age Group", list(AGE_LABELS), format_func=AGE_LABELS.get)
    gender = st.selectbox("Select Gender", list(GENDER_LABELS), format_func=GENDER_LABELS.get)

    predict_button = st.button("🔮 Make Prediction")

current_inputs = [ppq1, pv3, pv2, age, gender]

# Results are kept in session state so that nudging a slider after predicting does not
# wipe the chart; we just flag that it is now stale.
if predict_button and model is not None:
    with st.spinner("Updating prediction..."):
        try:
            predicted_class, probabilities = predict(model, current_inputs)
            st.session_state["result"] = {
                "inputs": current_inputs,
                "predicted_class": predicted_class,
                "probabilities": probabilities,
            }
        except Exception as e:
            st.session_state.pop("result", None)
            st.session_state["error"] = str(e)
        else:
            st.session_state.pop("error", None)

with col2:
    st.header("🔮 Prediction Results")

    result = st.session_state.get("result")
    if model is None:
        st.warning("⚠️ Model is not loaded correctly.")
    elif "error" in st.session_state:
        st.error(f"❗ Error during prediction: {st.session_state['error']}")
    elif result is None:
        st.info("Click the 'Make Prediction' button to see results!")
    else:
        st.success(
            CLASS_MESSAGES.get(
                result["predicted_class"],
                f"Prediction: class {result['predicted_class']}.",
            )
        )
        if result["inputs"] != current_inputs:
            st.caption("↻ Inputs have changed — click 'Make Prediction' to refresh.")
        st.plotly_chart(probability_chart(result["probabilities"]), width="stretch")
