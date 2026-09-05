"""Streamlit app predicting subsistence-retail purchase intention from survey inputs.

The model is a RandomForestClassifier trained on `Subsistence Retail Consumer Data.xlsx`
and shipped as `random_forest_model.pkl`. See train.py to regenerate it.

The interface is built to explain itself: it shows the real survey statements behind
each control, updates live so cause and effect are visible, reads every probability
against the survey's own base rate, and names the change that would move the outcome
most.
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

# How the 281 surveyed shoppers actually answered, from the training data
# (value_counts on the collapsed PI1 target: {0: 28, 1: 196, 2: 57}). Shown as a
# reference mark on every bar so a probability is read against the real base rate
# rather than against nothing: 70% of shoppers intended to buy, so a high
# "strong intention" score is less remarkable than it first looks.
SURVEY_N = 281
SURVEY_COUNTS = {0: 28, 1: 196, 2: 57}
SURVEY_RATE = {k: v / SURVEY_N for k, v in SURVEY_COUNTS.items()}

# The statements respondents actually rated, from the notebook that built the
# minimum dataset. Showing them turns abstract variable names into a question a
# person answered.
SURVEY_ITEMS = {
    "PPQ1": "“The overall quality of products I buy from the grocery store is good.”",
    "PV3": "“In this grocery store, compared to other stores outside the township, "
           "I can save money.”",
    "PV2": "“The grocery store products are affordable.”",
}

CLASS_BLURB = {
    0: "This shopper is unlikely to buy from the store.",
    1: "This shopper is likely to buy from the store.",
    2: "This shopper is undecided and could go either way.",
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

# Only the attitudinal answers are things a store can actually change. Age and
# gender are demographics, so they are deliberately excluded from the "what would
# move this" suggestion.
LEVERS = {
    "PPQ1": ("product quality", QUALITY_LABELS),
    "PV3": ("the sense of saving money here", SAVING_LABELS),
    "PV2": ("affordability", AFFORD_LABELS),
}

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
    margin: 0 0 0.35rem;
    line-height: 1.1;
}
.pi-sub {
    color: var(--ink-dim);
    font-size: 0.97rem;
    line-height: 1.55;
    margin: 0 0 1.8rem;
    max-width: 72ch;
}
.pi-sub b { color: var(--ink); font-weight: 600; }
.pi-card-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--ink-faint);
    margin: 0 0 0.35rem;
}
.pi-card-lede {
    font-size: 0.86rem;
    color: var(--ink-dim);
    margin: 0 0 1.3rem;
    line-height: 1.5;
}
.pi-step {
    display: inline-block;
    min-width: 1.35rem;
    height: 1.35rem;
    line-height: 1.35rem;
    text-align: center;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid var(--glass-edge);
    color: var(--ink);
    font-size: 0.72rem;
    margin-right: 0.5rem;
}

/* The survey statement behind each control. */
.pi-item {
    font-size: 0.82rem;
    color: var(--ink-faint);
    font-style: italic;
    line-height: 1.45;
    margin: -0.35rem 0 0.15rem;
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
    margin-bottom: 1.5rem;
}

/* Emphasis bars: the predicted class carries its hue, the rest recede.
   The tick is the survey's own rate for that outcome. */
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
    position: relative;
    height: 9px;
    background: rgba(255, 255, 255, 0.07);
    border-radius: 0 4px 4px 0;
}
.pi-fill { height: 100%; border-radius: 0 4px 4px 0; }
.pi-ref {
    position: absolute;
    top: -4px;
    bottom: -4px;
    width: 2px;
    background: rgba(255, 255, 255, 0.55);
    border-radius: 1px;
}
.pi-bar-val {
    font-size: 0.87rem;
    text-align: right;
    color: var(--ink-dim);
    font-variant-numeric: tabular-nums;
}
.pi-bar-val.is-lead { color: var(--ink); font-weight: 600; }

.pi-legend {
    font-size: 0.76rem;
    color: var(--ink-faint);
    margin-top: 0.9rem;
    line-height: 1.5;
}
.pi-legend .tick {
    display: inline-block;
    width: 2px;
    height: 0.72rem;
    background: rgba(255, 255, 255, 0.55);
    vertical-align: -2px;
    margin: 0 0.3rem 0 0.1rem;
}

/* The single change that would move the outcome most. */
.pi-lever {
    margin-top: 1.15rem;
    padding: 0.85rem 1rem;
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--glass-edge);
    font-size: 0.86rem;
    color: var(--ink-dim);
    line-height: 1.55;
}
.pi-lever b { color: var(--ink); font-weight: 600; }
.pi-lever .up { color: #45c98d; font-weight: 600; }

.pi-caveat {
    margin-top: 0.9rem;
    font-size: 0.8rem;
    color: #e0b341;
    line-height: 1.5;
}
.pi-foot {
    margin-top: 1.9rem;
    font-size: 0.78rem;
    color: var(--ink-faint);
    line-height: 1.6;
    max-width: 80ch;
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


def best_lever(model, inputs, base_strong):
    """The single answer change that would raise 'strong intention' the most.

    Every candidate profile is scored in one batched call rather than one call per
    variant. Returns None when nothing improves on the current profile.
    """
    if 1 not in list(model.classes_):
        return None
    position = {name: i for i, name in enumerate(FEATURE_ORDER)}

    rows, meta = [], []
    for feature, (phrase, labels) in LEVERS.items():
        i = position[feature]
        for value in labels:
            if value == inputs[i]:
                continue
            variant = list(inputs)
            variant[i] = value
            rows.append(variant)
            meta.append((phrase, labels[inputs[i]], labels[value]))
    if not rows:
        return None

    strong_col = list(model.classes_).index(1)
    scores = model.predict_proba(pd.DataFrame(rows, columns=FEATURE_ORDER))[:, strong_col]
    best_i = int(scores.argmax())
    gain = (scores[best_i] - base_strong) * 100
    if gain < 0.5:  # nothing meaningfully better than where we already are
        return None
    phrase, was, now = meta[best_i]
    return phrase, was, now, gain


def result_html(predicted_class, probabilities):
    """Hero figure plus one emphasis bar per class, ordered along the scale."""
    accent = CLASS_COLORS.get(predicted_class, "#8b7cf6")
    confidence = probabilities.get(predicted_class, 0.0) * 100

    rows = []
    for value in CLASS_DISPLAY_ORDER:
        if value not in probabilities:
            continue
        pct = probabilities[value] * 100
        ref = SURVEY_RATE.get(value, 0.0) * 100
        lead = value == predicted_class
        # Only the predicted class carries colour; the others stay recessive so the
        # answer is not buried in three competing hues.
        fill = CLASS_COLORS[value] if lead else "rgba(255,255,255,0.22)"
        cls = " is-lead" if lead else ""
        rows.append(
            '<div class="pi-bar" title="{name} — this profile {pct:.1f}%, '
            'survey {ref:.1f}%">'
            '<div class="pi-bar-name{cls}">{name}</div>'
            '<div class="pi-track">'
            '<div class="pi-fill" style="width:{pct:.1f}%;background:{fill}"></div>'
            '<div class="pi-ref" style="left:{ref:.1f}%"></div>'
            "</div>"
            '<div class="pi-bar-val{cls}">{pct:.1f}%</div>'
            "</div>".format(name=CLASS_LABELS[value], pct=pct, ref=ref, fill=fill, cls=cls)
        )

    return (
        '<div class="pi-eyebrow">The model expects</div>'
        '<div class="pi-answer" style="color:{accent}">{label}</div>'
        '<div class="pi-blurb">{blurb}</div>'
        '<div class="pi-hero">{conf:.1f}%</div>'
        '<div class="pi-hero-cap">of the forest\'s 100 trees voted this way</div>'
        "{rows}"
        '<div class="pi-legend"><span class="tick"></span>marks how the '
        "{n} surveyed shoppers actually answered, so you can see whether this "
        "profile is above or below the norm.</div>".format(
            accent=accent,
            label=CLASS_LABELS[predicted_class],
            blurb=CLASS_BLURB.get(predicted_class, ""),
            conf=confidence,
            rows="".join(rows),
            n=SURVEY_N,
        )
    )


st.markdown('<div class="pi-title">Purchase Intention Predictor</div>', unsafe_allow_html=True)
st.markdown(
    "<div class='pi-sub'>"
    f"<b>{SURVEY_N} shoppers</b> at township grocery stores were surveyed. A random "
    "forest learned, from just five of their answers, whether a shopper intends to buy "
    "again. Answer as one of those shoppers on the left and the expectation on the "
    "right moves as you go — no button, so you can feel which answers actually matter."
    "</div>",
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
        st.markdown(
            '<div class="pi-card-title"><span class="pi-step">1</span>The shopper</div>'
            '<div class="pi-card-lede">Three statements from the questionnaire, rated '
            "the way a respondent would, plus who they are.</div>",
            unsafe_allow_html=True,
        )

        st.markdown(f'<div class="pi-item">{SURVEY_ITEMS["PPQ1"]}</div>', unsafe_allow_html=True)
        ppq1 = st.select_slider(
            "Perceived product quality",
            options=list(QUALITY_LABELS),
            value=3,
            format_func=QUALITY_LABELS.get,
            key="ppq1",
        )
        st.markdown(f'<div class="pi-item">{SURVEY_ITEMS["PV3"]}</div>', unsafe_allow_html=True)
        pv3 = st.select_slider(
            "Can save money here",
            options=list(SAVING_LABELS),
            value=3,
            format_func=SAVING_LABELS.get,
            key="pv3",
        )
        st.markdown(f'<div class="pi-item">{SURVEY_ITEMS["PV2"]}</div>', unsafe_allow_html=True)
        pv2 = st.select_slider(
            "Affordability of the offering",
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

current_inputs = [ppq1, pv3, pv2, age, gender]

with col2:
    with st.container(border=True, key="prediction_card"):
        st.markdown(
            '<div class="pi-card-title"><span class="pi-step">2</span>Their intention</div>'
            '<div class="pi-card-lede">Answering: “I intend to purchase from this '
            'grocery store.”</div>',
            unsafe_allow_html=True,
        )

        if model is None:
            st.error("Model unavailable, so no prediction can be shown.")
        else:
            try:
                predicted_class, probabilities = predict(model, current_inputs)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
            else:
                st.markdown(result_html(predicted_class, probabilities), unsafe_allow_html=True)

                strong = probabilities.get(1, 0.0)
                lever = best_lever(model, current_inputs, strong)
                if lever:
                    phrase, was, now, gain = lever
                    st.markdown(
                        '<div class="pi-lever">Biggest lever: moving <b>{phrase}</b> from '
                        "<b>{was}</b> to <b>{now}</b> would raise the chance of buying by "
                        '<span class="up">+{gain:.1f} points</span>.</div>'.format(
                            phrase=phrase, was=was, now=now, gain=gain
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="pi-lever">No single answer change would raise the '
                        "chance of buying from here.</div>",
                        unsafe_allow_html=True,
                    )

                if predicted_class == 0:
                    st.markdown(
                        '<div class="pi-caveat">Read this one carefully: only '
                        f"{SURVEY_COUNTS[0]} of the {SURVEY_N} shoppers surveyed said no, "
                        "and the model catches about half of them. It is least reliable "
                        "on exactly this answer.</div>",
                        unsafe_allow_html=True,
                    )

                with st.expander("See the numbers"):
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Outcome": [CLASS_LABELS[v] for v in CLASS_DISPLAY_ORDER],
                                "This shopper": [
                                    "{:.1f}%".format(probabilities.get(v, 0.0) * 100)
                                    for v in CLASS_DISPLAY_ORDER
                                ],
                                "All surveyed": [
                                    "{:.1f}%".format(SURVEY_RATE[v] * 100)
                                    for v in CLASS_DISPLAY_ORDER
                                ],
                            }
                        ),
                        hide_index=True,
                        width="stretch",
                    )

st.markdown(
    '<div class="pi-foot">'
    f"How much to trust this: {SURVEY_COUNTS[1]} of the {SURVEY_N} surveyed shoppers "
    f"({SURVEY_RATE[1] * 100:.0f}%) intended to buy, so always answering “strong "
    "intention” would already be right most of the time. The model gets 86% of "
    "held-out shoppers right — better, but not by as much as a high percentage above "
    "suggests. It reads five answers only; price, stock and distance are not in it."
    "</div>",
    unsafe_allow_html=True,
)
