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
    "01": "01 - Ain",
    "02": "02 - Aisne",
    "03": "03 - Allier",
    "04": "04 - Alpes-de-Haute-Provence",
    "05": "05 - Hautes-Alpes",
    "06": "06 - Alpes-Maritimes",
    "07": "07 - Ardèche",
    "08": "08 - Ardennes",
    "09": "09 - Ariège",
    "10": "10 - Aube",
    "11": "11 - Aude",
    "12": "12 - Aveyron",
    "13": "13 - Bouches-du-Rhône",
    "14": "14 - Calvados",
    "15": "15 - Cantal",
    "16": "16 - Charente",
    "17": "17 - Charente-Maritime",
    "18": "18 - Cher",
    "19": "19 - Corrèze",

    "2A": "2A - Corse-du-Sud",
    "2B": "2B - Haute-Corse",

    "21": "21 - Côte-d'Or",
    "22": "22 - Côtes-d'Armor",
    "23": "23 - Creuse",
    "24": "24 - Dordogne",
    "25": "25 - Doubs",
    "26": "26 - Drôme",
    "27": "27 - Eure",
    "28": "28 - Eure-et-Loir",
    "29": "29 - Finistère",
    "30": "30 - Gard",
    "31": "31 - Haute-Garonne",
    "32": "32 - Gers",
    "33": "33 - Gironde",
    "34": "34 - Hérault",
    "35": "35 - Ille-et-Vilaine",
    "36": "36 - Indre",
    "37": "37 - Indre-et-Loire",
    "38": "38 - Isère",
    "39": "39 - Jura",
    "40": "40 - Landes",
    "41": "41 - Loir-et-Cher",
    "42": "42 - Loire",
    "43": "43 - Haute-Loire",
    "44": "44 - Loire-Atlantique",
    "45": "45 - Loiret",
    "46": "46 - Lot",
    "47": "47 - Lot-et-Garonne",
    "48": "48 - Lozère",
    "49": "49 - Maine-et-Loire",
    "50": "50 - Manche",
    "51": "51 - Marne",
    "52": "52 - Haute-Marne",
    "53": "53 - Mayenne",
    "54": "54 - Meurthe-et-Moselle",
    "55": "55 - Meuse",
    "56": "56 - Morbihan",
    "57": "57 - Moselle",
    "58": "58 - Nièvre",
    "59": "59 - Nord",
    "60": "60 - Oise",
    "61": "61 - Orne",
    "62": "62 - Pas-de-Calais",
    "63": "63 - Puy-de-Dôme",
    "64": "64 - Pyrénées-Atlantiques",
    "65": "65 - Hautes-Pyrénées",
    "66": "66 - Pyrénées-Orientales",
    "67": "67 - Bas-Rhin",
    "68": "68 - Haut-Rhin",
    "69": "69 - Rhône",
    "70": "70 - Haute-Saône",
    "71": "71 - Saône-et-Loire",
    "72": "72 - Sarthe",
    "73": "73 - Savoie",
    "74": "74 - Haute-Savoie",
    "75": "75 - Paris",
    "76": "76 - Seine-Maritime",
    "77": "77 - Seine-et-Marne",
    "78": "78 - Yvelines",
    "79": "79 - Deux-Sèvres",
    "80": "80 - Somme",
    "81": "81 - Tarn",
    "82": "82 - Tarn-et-Garonne",
    "83": "83 - Var",
    "84": "84 - Vaucluse",
    "85": "85 - Vendée",
    "86": "86 - Vienne",
    "87": "87 - Haute-Vienne",
    "88": "88 - Vosges",
    "89": "89 - Yonne",
    "90": "90 - Territoire de Belfort",
    "91": "91 - Essonne",
    "92": "92 - Hauts-de-Seine",
    "93": "93 - Seine-Saint-Denis",
    "94": "94 - Val-de-Marne",
    "95": "95 - Val-d'Oise",
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
    surface = st.selectbox("Surface condition", ["Normal", "Wet", "Snow or ice", "Other"], format_func=lambda value: {
        "Other": "Other hazardous surface (mud, oil, etc...)"
    }.get(value, value))
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
        prediction_context = f"conditional on department {department_selection}"
    else:
        prediction_context = "without any department effect"

    st.markdown(f"""
    <div class="prediction-card">
        <div style="font-size: 1.1rem;">PROBABILITY OF SERIOUS INJURY OR FATALITY</div>
        <div class="prediction-value">{probability_percentage:.2f}%</div>
        <div class="prediction-ci">95% CI: [{ci_lower_percentage:.2f}%, {ci_upper_percentage:.2f}%]</div>
        <div style="font-size: 0.9rem; margin-top: 0.5rem;">{prediction_context}, given that a person is involved in the accident</div>
    </div>
    """, unsafe_allow_html=True)

