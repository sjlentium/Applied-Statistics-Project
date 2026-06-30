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
    page_title="Accident Severity Risk",
    page_icon="🚗",
    initial_sidebar_state="collapsed" # Collapsed by default for mobile focus
)

# Keep sidebar for desktop users, but duplicate vital info for mobile
with st.sidebar:
    st.markdown("### 📊 Applied Statistics | Group 27")
    st.markdown("**Authors:** F. Cola, M. Filoramo, G. Genouville, V. Mariani")
    st.markdown("**Tutor:** S. Panzeri")
    st.divider()
    st.markdown("""
    ### 🛣️ **Statistical Modeling of Traffic Accident Data**
    This application predicts the probability that a person involved in a road accident is seriously injured or killed.
    """)

# Mobile-friendly CSS with Media Queries
# - Base styles for desktop/tablet
# - @media query strictly for mobile screens to scale down fonts and padding
st.markdown("""
    <style>
        .stButton button { 
            font-size: 1.2rem; 
            background-color: #4CAF50; 
            color: white; 
            border-radius: 12px; 
            border: none;
        }
        .prediction-card { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 2rem; 
            border-radius: 20px; 
            color: white; 
            text-align: center; 
            margin: 1rem 0; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .prediction-value { 
            font-size: 3rem; 
            font-weight: bold; 
            margin: 0.5rem 0; 
        }
        .prediction-ci { 
            font-size: 1.2rem; 
            opacity: 0.9; 
            margin-top: 0.25rem; 
        }
        
        /* Mobile Specific Styling */
        @media (max-width: 768px) {
            .prediction-card {
                padding: 1.25rem;
                border-radius: 15px;
            }
            .prediction-value {
                font-size: 2.2rem; /* Scaled down for narrow screens */
            }
            .prediction-ci {
                font-size: 1rem;
            }
            .prediction-context {
                font-size: 0.8rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Directory pointing
DATA_DIR = Path(__file__).resolve().parent / "DATA_DIR"

@st.cache_resource
def load_model():
    """
    Load the fitted GLMM coefficients and the bootstrap draws exported from R.
    (Data loading logic remains unchanged to ensure statistical fidelity).
    """
    try:
        point_fixed_effects = pd.read_csv(DATA_DIR / "point_fixed_effects.csv").set_index("term")["estimate"]
        bootstrap_fixed_effects = pd.read_csv(DATA_DIR / "bootstrap_fixed_effects.csv.gz").drop(columns="bootstrap_id")
        point_random_effects = pd.read_csv(DATA_DIR / "point_random_effects.csv", dtype={"department_code": str}).set_index("department_code")["estimate"]
        bootstrap_random_effects = pd.read_csv(DATA_DIR / "bootstrap_random_effects.csv.gz").drop(columns="bootstrap_id")
        
        return {
            "point": point_fixed_effects,
            "bootstrap": bootstrap_fixed_effects,
            "point_department": point_random_effects,
            "bootstrap_department": bootstrap_random_effects,
        }
    except FileNotFoundError:
        st.error("Model data not found. Please ensure the DATA_DIR contains the required CSV files.")
        st.stop()

model = load_model()

# Dictionary and label logic remains unchanged
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
    if level == REFERENCE_LEVELS[variable]:
        return None
    return f"{variable}{level}"

DEPARTMENT_LABELS = {
    # (Truncated for brevity in the snippet, insert your full dict here)
    "01": "01 - Ain", "02": "02 - Aisne", "75": "75 - Paris" 
}

def department_label(code):
    return DEPARTMENT_LABELS.get(code, code)

UNKNOWN_DEPARTMENT = "Unknown / average department"
department_codes = model["point_department"].index.tolist()
department_options = [UNKNOWN_DEPARTMENT] + [department_label(code) for code in department_codes]
department_code_by_label = {department_label(code): code for code in department_codes}

# --- UI LAYOUT STARTS HERE ---

st.title("🚗 **Serious-Injury Predictor**")
st.markdown("*GLMM prediction with 95% bootstrap CI (France)*")

# Context expander for mobile users who don't see the sidebar
with st.expander("ℹ️ About this model"):
    st.write("Using a Generalized Linear Mixed Model (GLMM) fitted on the French national road accident database, this calculates the logistic probability of severe injury or fatality based on road and environmental conditions.")

# Instead of forced columns, use sequential expanders/sections.
# This ensures inputs utilize the full width of mobile screens without squishing.

st.subheader("📍 Location")
department_selection = st.selectbox("Department (optional)", department_options)

# Grouping inputs logically saves vertical space on mobile and avoids scroll-fatigue
with st.expander("🌤️ Environmental Conditions", expanded=True):
    weather = st.selectbox("Weather", ["Normal", "Drizzle", "Snow or hail", "Smoke or fog", "Dazzling", "Overcast", "Storm or heavy rain", "Other"])
    brightness = st.selectbox("Brightness", ["Broad daylight", "Dusk or dawn", "Night with street lighting on", "Night without street lighting"])
    urban = st.selectbox("Area type", ["Urban", "Rural"])

with st.expander("🛣️ Road Infrastructure", expanded=True):
    intersection = st.selectbox("Intersection", ["No intersection", "X-shaped", "T-shaped", "Y-shaped", "4 or more branches", "Roundabout", "Square", "Other intersection"])
    traffic = st.selectbox("Traffic pattern", ["Two-way", "One-way", "Dual carriageway", "Dynamical lane assignment"])
    surface = st.selectbox("Surface condition", ["Normal", "Wet", "Snow or ice", "Other"], format_func=lambda value: {"Other": "Other hazardous surface (mud, oil, etc...)"}.get(value, value))
    turn = st.selectbox("Road layout (Turn)", ["Straight", "Left bend", "Right bend", "S-shaped bend"])
    gradient = st.selectbox("Road steepness", ["Flat", "Hill", "Top of the hill", "Bottom of the hill"])

st.divider()

# use_container_width=True replaces the complex column math.
# It makes the button span perfectly edge-to-edge on mobile, and fills the container on desktop.
predict_button = st.button("🔮 **PREDICT PROBABILITY**", use_container_width=True)

if predict_button:
    
    # Selection mapping
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

    active_terms = [
        term_name(variable, level) for variable, level in selections.items() if term_name(variable, level) is not None
    ]

    point_coefficients = model["point"]
    missing_point_terms = [term for term in active_terms if term not in point_coefficients.index]
    
    if missing_point_terms:
        st.error("Some selected levels do not match the coefficient names exported from R.")
        st.stop()
    
    # 1. Point prediction
    eta_point = point_coefficients["(Intercept)"] + point_coefficients[active_terms].sum()

    department_chosen = department_selection != UNKNOWN_DEPARTMENT
    if department_chosen:
        department_code = department_code_by_label[department_selection]
        eta_point += model["point_department"][department_code]

    probability_point = 1 / (1 + np.exp(-eta_point))

    # 2. 95% confidence interval
    bootstrap_coefficients = model["bootstrap"]
    eta_bootstrap = bootstrap_coefficients["(Intercept)"] + bootstrap_coefficients[active_terms].sum(axis=1)
    
    if department_chosen:
        eta_bootstrap = eta_bootstrap + model["bootstrap_department"][department_code]

    probability_bootstrap = 1 / (1 + np.exp(-eta_bootstrap))
    ci_lower, ci_upper = np.percentile(probability_bootstrap, [2.5, 97.5])

    probability_percentage = probability_point * 100
    ci_lower_percentage = ci_lower * 100
    ci_upper_percentage = ci_upper * 100

    prediction_context = f"conditional on department {department_selection}" if department_chosen else "without any department effect"

    # The prediction card relies on the @media CSS to ensure it renders correctly on mobile
    st.markdown(f"""
    <div class="prediction-card">
        <div style="font-size: 1.1rem; font-weight: 500;">PROBABILITY OF SERIOUS INJURY OR FATALITY</div>
        <div class="prediction-value">{probability_percentage:.2f}%</div>
        <div class="prediction-ci">95% CI: [{ci_lower_percentage:.2f}%, {ci_upper_percentage:.2f}%]</div>
        <div class="prediction-context" style="margin-top: 0.75rem; font-style: italic;">{prediction_context}, given that a person is involved in the accident</div>
    </div>
    """, unsafe_allow_html=True)
    
