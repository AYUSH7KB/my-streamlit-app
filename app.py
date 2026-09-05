import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------------
# 1. PAGE SETUP & HEADER
# -------------------------------------------------------------
st.set_page_config(page_title="LifeSim Prototype", layout="wide")

st.title("🧬 LifeSim: AI-Driven Health Projection")
st.caption("Capstone Project Demo • Class XII Artificial Intelligence")

# -------------------------------------------------------------
# 2. MACHINE LEARNING ENGINE (Scikit-learn)
# -------------------------------------------------------------
# NOTE ON THE FIX: the original model was trained on only 8 hand-picked points.
# That caused two problems:
#   (a) it overfit and extrapolated to nonsense (negative ages) at slider extremes
#   (b) daily_steps was correlated with resting_hr in the tiny dataset, so the
#       model learned to almost ignore steps -- even though the UI text claimed
#       steps mattered a lot.
#
# Fix: generate a larger synthetic training set that (1) covers the *entire*
# slider range so the model never has to extrapolate, and (2) varies each
# feature independently so the model learns a genuine, separate effect for
# steps, sleep, and resting HR.
@st.cache_resource
def train_biological_age_model():
    rng = np.random.default_rng(42)

    def true_bio_age(age, steps, sleep, hr):
        # Domain-informed relationship used only to generate realistic training
        # labels. Coefficients are illustrative, not clinically derived.
        return (
            age
            + 0.00035 * (7000 - steps)   # more steps -> lower biological age
            - 1.3 * (sleep - 7.0)        # more sleep -> lower biological age
            + 0.18 * (hr - 65)           # higher resting HR -> higher biological age
        )

    n = 300
    ages  = rng.uniform(15, 80, n)
    steps = rng.uniform(1000, 16000, n)
    sleep = rng.uniform(4.0, 10.0, n)
    hr    = rng.uniform(50, 105, n)
    noise = rng.normal(0, 1.5, n)  # measurement noise so the fit isn't a perfect line

    X_train = np.column_stack([ages, steps, sleep, hr])
    y_train = true_bio_age(ages, steps, sleep, hr) + noise
    y_train = np.clip(y_train, 0, 110)

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = train_biological_age_model()

MIN_BIO_AGE, MAX_BIO_AGE = 0.0, 110.0

def predict_bio_age(age, steps, sleep, hr):
    """Predict biological age and clamp to a physically sensible range."""
    raw = model.predict(np.array([[age, steps, sleep, hr]]))[0]
    return float(np.clip(raw, MIN_BIO_AGE, MAX_BIO_AGE))

# -------------------------------------------------------------
# 3. USER INPUT DASHBOARD (Sidebar)
# -------------------------------------------------------------
st.sidebar.header("User Lifestyle Parameters")

chronological_age = st.sidebar.slider("Chronological Age", 15, 80, 18)
daily_steps = st.sidebar.slider("Daily Steps", 1000, 16000, 4500, step=500)
sleep_hours = st.sidebar.slider("Sleep Duration (Hours)", 4.0, 10.0, 6.0, step=0.5)
resting_hr = st.sidebar.slider("Resting Heart Rate (BPM)", 50, 105, 76)

# -------------------------------------------------------------
# 4. PREDICTION & METRIC COMPARISONS
# -------------------------------------------------------------
predicted_bio_age = predict_bio_age(chronological_age, daily_steps, sleep_hours, resting_hr)
age_difference = predicted_bio_age - chronological_age

col1, col2, col3 = st.columns(3)
col1.metric("Chronological Age", f"{chronological_age} yrs")
col2.metric(
    "Predicted Biological Age",
    f"{predicted_bio_age:.1f} yrs",
    delta=f"{age_difference:+.1f} yrs",
    delta_color="inverse"
)
col3.metric(
    "Vitality Status",
    "Optimal" if age_difference <= 0 else "Accelerated Aging"
)

st.divider()

# -------------------------------------------------------------
# 5. FUTURE TWIN TRAJECTORY (+10, +20, +30 Years)
# -------------------------------------------------------------
st.subheader("📈 Future Twin Projection: Current vs. Optimized Path")

years = np.array([0, 10, 20, 30])

# Current path: same habits held constant, chronological age advances each decade.
current_trajectory = np.array([
    predict_bio_age(min(chronological_age + y, 80), daily_steps, sleep_hours, resting_hr)
    for y in years
])

# Optimized path: gradual habit improvements (8,500 steps, 7.5 hrs sleep, 65 bpm),
# with chronological age also advancing each decade.
optimized_trajectory = np.array([
    predict_bio_age(min(chronological_age + y, 80), 8500, 7.5, 65)
    for y in years
])

chart_data = pd.DataFrame({
    "Timeline (Years From Today)": [f"+{y} yrs" for y in years],
    "Current Lifestyle Trajectory": current_trajectory,
    "Optimized Habits Trajectory": optimized_trajectory
}).set_index("Timeline (Years From Today)")

st.line_chart(chart_data)

# -------------------------------------------------------------
# 6. ACTIONABLE AI RECOMMENDATIONS (now derived from the model itself)
# -------------------------------------------------------------
st.subheader("💡 Targeted Interventions")
rec_col1, rec_col2 = st.columns(2)

# Instead of hardcoded numbers, ask the model what actually changes if this one
# habit were improved, holding everything else constant. This keeps the copy
# honest and consistent with what the model has learned.
TARGET_STEPS = 8000
TARGET_SLEEP = 7.5

steps_gain = predicted_bio_age - predict_bio_age(
    chronological_age, TARGET_STEPS, sleep_hours, resting_hr
)
sleep_gain = predicted_bio_age - predict_bio_age(
    chronological_age, daily_steps, TARGET_SLEEP, resting_hr
)

with rec_col1:
    if daily_steps < TARGET_STEPS:
        st.warning(
            f"⚠️ **Step Deficit**: Increasing daily steps to {TARGET_STEPS:,} could reduce "
            f"predicted biological age by roughly {max(steps_gain, 0):.1f} years, per the model."
        )
    else:
        st.success("✅ **Activity Level**: Daily step goal meets the longevity baseline.")

with rec_col2:
    if sleep_hours < TARGET_SLEEP:
        st.warning(
            f"⚠️ **Sleep Deprivation**: Reaching {TARGET_SLEEP:.1f} hours of sleep could reduce "
            f"predicted biological age by roughly {max(sleep_gain, 0):.1f} years, per the model."
        )
    else:
        st.success("✅ **Rest & Recovery**: Sleep schedule supports optimal cell repair.")

st.caption(
    "Note: biological ages and recommendations are generated from a simple illustrative "
    "regression model trained on synthetic data, not real clinical measurements."
)
