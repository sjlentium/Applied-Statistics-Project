# streamlit_app.py - Accident Severity Predictor using GLMM Model
# Authors: F. Cola, M. Filoramo, G. Genouville, V. Mariani
# Tutor: Simone Panzeri
# Project: Statistical Modeling of Traffic Accident Data on Large Road Networks

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Accident Severity Risk Predictor",
    page_icon="🚗",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.markdown("### 📊 Applied Statistics | Group 27")
    st.markdown("**Authors:** F. Cola, M. Filoramo, G. Genouville, V. Mariani")
    st.markdown("**Tutor:** S. Panzeri")
    st.divider()
    st.markdown("""
    ### 🛣️ **Statistical Modeling of Traffic Accident Data on Large Road Networks**
    
    This application predicts the probability that a person involved in a road accident is seriously injured or killed.
    
    Using a Generalized Linear Mixed Model (GLMM) fitted on the French national road accident database, this tool calculates the logistic probability based on:
    - **Road infrastructure** (intersection type, traffic regime, turn, surface)
    - **Environmental conditions** (urban/rural setting, weather, brightness, gradient)
    
    The 95% confidence interval is obtained from a conditional parametric bootstrap (500 replications) of the fitted model.
    """)

# Mobile-friendly CSS
st.markdown("""
    <style>
        .stButton button { width: 100%; padding: 0.75rem; font-size: 1.2rem; background-color: #4CAF50; color: white; border-radius: 12px; }
        .stSelectbox, .stSlider, .stTimeInput { margin-bottom: 1rem; }
        .prediction-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 20px; color: white; text-align: center; margin: 1rem 0; }
        .prediction-value { font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0; }
        .prediction-ci { font-size: 1.1rem; opacity: 0.9; margin-top: 0.25rem; }
    </style>
""", unsafe_allow_html=True)

# Directory containing the files exported by the R bootstrap script:
# point_fixed_effects.csv, bootstrap_fixed_effects.csv.gz,
# point_random_effects.csv, bootstrap_random_effects.csv.gz.
# Adjust if the files live elsewhere.
DATA_DIR = Path(__file__).resolve().parent / "DATA_DIR"


@st.cache_resource
def load_model():
    """
    Load the fitted GLMM coefficients and the bootstrap draws exported
    from R (see model_metadata.json / point_fixed_effects.csv /
    bootstrap_fixed_effects.csv.gz / point_random_effects.csv /
    bootstrap_random_effects.csv.gz).

    Returns
    -------
    dict with:
      - 'point': pandas Series of fixed-effect point estimates, indexed by
                 term name (e.g. "weatherDrizzle", "urbanRural", ...).
      - 'bootstrap': DataFrame with one row per bootstrap replication and
                      one column per fixed-effect term (same columns as
                      'point', no "bootstrap_id" column).
      - 'point_department': pandas Series of department random-intercept
                             point estimates, indexed by department code.
      - 'bootstrap_department': DataFrame with one row per bootstrap
                                 replication and one column per department
                                 code (no "bootstrap_id" column).
    """
    point_fixed_effects = pd.read_csv(
        DATA_DIR / "point_fixed_effects.csv"
    ).set_index("term")["estimate"]

    bootstrap_fixed_effects = pd.read_csv(
        DATA_DIR / "bootstrap_fixed_effects.csv.gz"
    ).drop(columns="bootstrap_id")

    point_random_effects = pd.read_csv(
        DATA_DIR / "point_random_effects.csv",
        dtype={"department_code": str},
    ).set_index("department_code")["estimate"]

    bootstrap_random_effects = pd.read_csv(
        DATA_DIR / "bootstrap_random_effects.csv.gz"
    ).drop(columns="bootstrap_id")

    return {
        "point": point_fixed_effects,
        "bootstrap": bootstrap_fixed_effects,
        "point_department": point_random_effects,
        "bootstrap_department": bootstrap_random_effects,
    }


model = load_model()

# Maps each UI label to the exact column-name prefix used in the
# exported coefficients (must match design_matrix_columns in
# model_metadata.json, e.g. "weather" + "Drizzle" -> "weatherDrizzle").
REFERENCE_LEVELS = {
    "brightness": "Broad daylight",
    "urban": "Urban",
    "intersection": "No intersection",
    "weather": "Normal",
    "surface": "Normal",
    "traffic": "Two-way",
    "gradient": "Flat",
    "turn": "Straight",
}


def term_name(variable, level):
    """Build the coefficient name for a given variable/level pair.
    Returns None for the reference level, which contributes 0 by
    construction under treatment contrasts."""
    if level == REFERENCE_LEVELS[variable]:
        return None
    return f"{variable}{level}"


DEPARTMENT_LABELS = {
    "75": "75 - Paris",
    "977": "977 - Saint-Martin",
    "978": "978 - Saint-Barthélemy",
    "986": "986 - Wallis-et-Futuna",
}


def department_label(code):
    return DEPARTMENT_LABELS.get(code, code)


# "Unknown / average department" lets the user get a marginal prediction
# (no department-specific random intercept), which also widens the
# confidence interval to reflect the extra between-department uncertainty.
UNKNOWN_DEPARTMENT = "Unknown / average department"
department_codes = model["point_department"].index.tolist()
department_options = [UNKNOWN_DEPARTMENT] + [
    department_label(code) for code in department_codes
]
department_code_by_label = {
    department_label(code): code for code in department_codes
}


st.title("🚗 **Serious-Injury Risk Predictor**")
st.markdown("*GLMM prediction of injury probability with 95% bootstrap confidence interval (Metropolitan France & Corsica)*")

col1, col2 = st.columns(2)

with col1:

    department_selection = st.selectbox("Department (optional)", department_options)
    weather = st.selectbox("Weather", ["Normal", "Drizzle", "Snow or hail", "Smoke or fog", "Dazzling", "Overcast", "Storm or heavy rain", "Other"])
    brightness = st.selectbox("Brightness", ["Broad daylight", "Dusk or dawn", "Night with street lighting on", "Night without street lighting"])
    intersection = st.selectbox("Intersection",
                                ["No intersection", "X-shaped", "T-shaped", "Y-shaped", "4 or more branches",
                                 "Roundabout", "Square", "Other intersection"])
    traffic = st.selectbox("Traffic pattern",
                           ["Two-way", "One-way", "Dual carriageway", "Dynamical lane assignment"])

with col2:
    
    urban = st.selectbox("Area type", ["Urban", "Rural"])
    surface = st.selectbox("Surface condition", ["Normal", "Wet", "Snow or ice", "Other hazardous surface (mud, oil, etc...)"])
    turn = st.selectbox("Road layout (Turn)", ["Straight", "Left bend", "Right bend", "S-shaped bend"])
    gradient = st.selectbox("Road steepness", ["Flat", "Hill", "Top of the hill", "Bottom of the hill"])

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🔮 **PREDICT PROBABILITY**")

if predict_button:

    # Collect the (variable, selected level) pairs from the form above.
    selections = {
        "weather": weather,
        "brightness": brightness,
        "urban": urban,
        "intersection": intersection,
        "traffic": traffic,
        "surface": surface,
        "turn": turn,
        "gradient": gradient,
    }

    # Active coefficient names for the chosen levels (reference levels
    # are skipped, since they contribute 0 to the linear predictor).
    active_terms = [
        term_name(variable, level)
        for variable, level in selections.items()
        if term_name(variable, level) is not None
    ]

    # 1. Point prediction: sum the intercept, the active coefficients,
    # and (if a department was chosen) its random-intercept point estimate.
    point_coefficients = model["point"]

    missing_point_terms = [
        term
        for term in active_terms
        if term not in point_coefficients.index
    ]
    
    if missing_point_terms:
        st.error(
            "Some selected levels do not match the coefficient names "
            "exported from R."
        )
    
        st.write("Selected terms:", active_terms)
        st.write("Missing terms:", missing_point_terms)
    
        with st.expander("Available fixed-effect coefficients"):
            st.code("\n".join(point_coefficients.index.astype(str)))
    
        st.stop()
    
    eta_point = (
        point_coefficients["(Intercept)"]
        + point_coefficients[active_terms].sum()
    )

    department_chosen = department_selection != UNKNOWN_DEPARTMENT
    if department_chosen:
        department_code = department_code_by_label[department_selection]
        eta_point += model["point_department"][department_code]

    probability_point = 1 / (1 + np.exp(-eta_point))

    # 2. 95% confidence interval: repeat the same linear predictor for
    # every bootstrap replication, then take the empirical probability
    # at each replication and read off the 2.5th/97.5th percentiles.
    #
    # When no department is chosen, only the fixed-effect uncertainty is
    # propagated (marginal prediction). When a department is chosen, its
    # bootstrap random-intercept draws are added too, so the interval
    # reflects the full conditional uncertainty for that department.
    bootstrap_coefficients = model["bootstrap"]
    eta_bootstrap = (
        bootstrap_coefficients["(Intercept)"] + bootstrap_coefficients[active_terms].sum(axis=1)
    )
    if department_chosen:
        eta_bootstrap = eta_bootstrap + model["bootstrap_department"][department_code]

    probability_bootstrap = 1 / (1 + np.exp(-eta_bootstrap))

    ci_lower, ci_upper = np.percentile(probability_bootstrap, [2.5, 97.5])

    probability_percentage = probability_point * 100
    ci_lower_percentage = ci_lower * 100
    ci_upper_percentage = ci_upper * 100

    if department_chosen:
        prediction_context = f"Conditional on department {department_selection}"
    else:
        prediction_context = "Marginal prediction (averaged over departments)"

    st.markdown(f"""
    <div class="prediction-card">
        <div style="font-size: 1.1rem;">PROBABILITY OF SERIOUS INJURY OR FATALITY</div>
        <div class="prediction-value">{probability_percentage:.2f}%</div>
        <div class="prediction-ci">95% CI: [{ci_lower_percentage:.2f}%, {ci_upper_percentage:.2f}%]</div>
        <div style="font-size: 0.9rem; margin-top: 0.5rem;">{prediction_context}, given that a person is involved in the accident</div>
    </div>
    """, unsafe_allow_html=True)

