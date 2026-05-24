# streamlit_app.py - Accident Severity Predictor using GLM Model
# Authors: F. Cola, M. Filoramo, G. Genouville, V. Mariani
# Supervisor: Simone Panzeri
# Project: Statistical Modeling of Traffic Accident Data on Large Road Networks

import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import time
import plotly.express as px
import os
import sys

# Page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="Accident Severity Predictor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Mobile-friendly CSS
st.markdown("""
    <style>
        .stButton button {
            width: 100%;
            padding: 0.75rem;
            font-size: 1.2rem;
            background-color: #4CAF50;
            color: white;
            border-radius: 12px;
        }
        .stSelectbox, .stSlider, .stTimeInput {
            margin-bottom: 1rem;
        }
        div[data-testid="stToolbar"] {
            display: none;
        }
        @media (max-width: 768px) {
            .main > div {
                padding: 1rem;
            }
            h1 {
                font-size: 1.8rem !important;
            }
        }
        .prediction-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 20px;
            color: white;
            text-align: center;
            margin: 1rem 0;
        }
        .prediction-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        .info-text {
            color: #666;
            font-size: 0.8rem;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Introduction and project description
st.markdown("""
## 📊 **Statistical Modeling of Traffic Accident Data on Large Road Networks**

This application is developed as part of a research project on **Statistical Modeling of Traffic Accident Data on Large Road Networks**, conducted in collaboration with public institutions. The goal is to gain an informed view of how traffic accidents are distributed over space and time, and how accident severity relates to environmental and contextual variables such as road characteristics, day category, and weather conditions.

Using a Generalized Linear Model (GLM) fitted on the French national road accident database (2019-2024), this tool predicts the number of people involved in an accident based on:
- **Temporal factors** (year, time, day type, school holidays)
- **Road infrastructure** (intersection type, traffic regime, gradient, speed limits)
- **Environmental conditions** (urban/rural setting, illuminance)

The model integrates spatial intensity and severity patterns to identify accident clusters and support road safety decision-making.
""")

st.divider()

# Load the pre-trained GLM model
@st.cache_resource
def load_model():
    """Load the pre-trained GLM model coefficients"""
    
    # Model coefficients from GLM analysis of French accident data (2019-2024)
    model_coefficients = {
        'intercept': 2.648e+00,
        'source_year': {
            '2020': -5.033e-02,  # COVID-19 lockdown effect
            '2021': 3.078e-02,
            '2022': 4.399e-02,
            '2023': 5.182e-02,
            '2024': 6.130e-02
        },
        'numeric_time': 2.335e-01,  # Normalized hour (0-1)
        'urban': {
            'Urban': 8.480e-02,
            'Non-urban': 0.0  # reference
        },
        'intersection': {
            'X-shaped': 2.876e-01,
            'T-shaped': 1.566e-01,
            'Y-shaped': 1.073e-01,
            '4 or more branches': 2.041e-01,
            'Roundabout': -5.538e-02,
            'Square': 4.284e-02,
            'Level crossing': 2.501e-01,
            'Other intersection': 1.076e-01,
            'Unknown': 2.095e-03,
            'No intersection': 0.0  # reference
        },
        'traffic': {
            'One-way': 6.949e-02,
            'Two-way': 2.322e-03,
            'Dual carriageway': 1.814e-01,
            'Dynamical lane assignment': 1.270e-01,
            'Other/Unknown': 0.0  # reference
        },
        'gradient': {
            'Flat': -1.138e+00,
            'Hill': -1.218e+00,
            'Top of the hill': -1.131e+00,
            'Bottom of the hill': -1.232e+00,
            'Other/Unknown': 0.0  # reference
        },
        'max_speed': 8.843e-03,
        'school_holidays': 4.019e-02,
        'day_type': {
            'Work day': -1.388e-01,
            'Saturday': 7.037e-03,
            'Sunday': 0.0  # reference
        },
        'illuminance': 9.131e-07
    }
    
    return model_coefficients

# Load model
model = load_model()

# Title
st.title("🚗 **Accident Severity Predictor**")
st.markdown("*GLM-based prediction of people involved in road accidents (French database 2019-2024)*")

# Layout columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("📅 **Time & Date**")
    
    # Source year
    source_year = st.selectbox(
        "Year",
        options=[2020, 2021, 2022, 2023, 2024],
        help="Year of the accident (data available from French national database)"
    )
    
    # Numeric time (converted from time input)
    accident_time = st.time_input(
        "Accident time",
        value=time(12, 0),
        help="Time when the accident occurred (24-hour format)"
    )
    numeric_time = accident_time.hour + accident_time.minute/60
    # Normalize to 0-1 range (assuming 24-hour cycle)
    numeric_time_normalized = numeric_time / 24
    
    # School holidays
    school_holidays = st.selectbox(
        "School holidays",
        options=["No", "Yes"],
        help="School holidays in effect (data from French Ministry of Education)"
    )
    school_holidays_bool = school_holidays == "Yes"
    
    # Day type
    day_type = st.selectbox(
        "Day type",
        options=["Work day", "Saturday", "Sunday"],
        help="Type of day (workday vs weekend)"
    )

    # Illuminance
    illuminance = st.slider(
        "Illuminance (lux)",
        min_value=0,
        max_value=100000,
        value=5000,
        step=1000,
        help="Light level at accident location. Computed from solar altitude via skylight R package (0 = darkness, 100000 = bright sunlight)"
    )

with col2:
    st.subheader("🏗️ **Road & Location**")
    
    # Urban
    urban = st.selectbox(
        "Area type",
        options=["Non-urban", "Urban"],
        help="Urban (code 2) or rural (code 1) area as per French classification"
    )
    
    # Intersection type
    intersection = st.selectbox(
        "Intersection type",
        options=["No intersection", "X-shaped", "T-shaped", "Y-shaped", 
                 "4 or more branches", "Roundabout", "Square", 
                 "Level crossing", "Other intersection", "Unknown"],
        help="Type of intersection at accident location (French classification)"
    )
    
    # Traffic pattern
    traffic = st.selectbox(
        "Traffic pattern",
        options=["Other/Unknown", "One-way", "Two-way", "Dual carriageway", 
                 "Dynamical lane assignment"],
        help="Traffic regime at accident location"
    )
    
    # Gradient
    gradient = st.selectbox(
        "Road gradient",
        options=["Other/Unknown", "Flat", "Hill", "Top of the hill", "Bottom of the hill"],
        help="Declivity of the road at accident location"
    )
    
    # Max speed
    max_speed = st.slider(
        "Speed limit (km/h)",
        min_value=20,
        max_value=130,
        value=50,
        step=10,
        help="Maximum allowed speed on the road (speed limit)"
    )

# Prediction button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🔮 **PREDICT**", use_container_width=True)

# Display prediction
if predict_button:
    # Calculate prediction
    prediction = model['intercept']
    prediction += model['source_year'].get(str(source_year), 0)
    prediction += model['numeric_time'] * numeric_time_normalized
    if urban == "Urban":
        prediction += model['urban']['Urban']
    prediction += model['intersection'].get(intersection, 0)
    prediction += model['traffic'].get(traffic, 0)
    prediction += model['gradient'].get(gradient, 0)
    prediction += model['max_speed'] * max_speed
    if school_holidays_bool:
        prediction += model['school_holidays']
    prediction += model['day_type'].get(day_type, 0)
    prediction += model['illuminance'] * illuminance
    
    # Ensure non-negative prediction
    prediction = max(0, prediction)
    
    # Display prediction
    st.markdown(f"""
    <div class="prediction-card">
        <div style="font-size: 1.1rem;">📊 PREDICTED NUMBER OF PEOPLE INVOLVED</div>
        <div class="prediction-value">{prediction:.1f}</div>
        <div style="font-size: 0.9rem;">people (including injured, hospitalized, and killed)</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show input details
    with st.expander("📋 **Input details**"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("📅 Year", source_year)
            st.metric("🕐 Time", f"{accident_time.strftime('%H:%M')}")
            st.metric("📆 Day type", day_type)
        with col_b:
            st.metric("🏙️ Area", urban)
            st.metric("🛣️ Speed limit", f"{max_speed} km/h")
            st.metric("🎒 School holidays", school_holidays)
        with col_c:
            st.metric("🔄 Intersection", intersection)
            st.metric("🚦 Traffic", traffic)
            st.metric("⛰️ Gradient", gradient)
    
    # Show model coefficients
    with st.expander("🔬 **Model coefficients**"):
        coef_data = []
        
        coef_data.append(["Intercept", f"{model['intercept']:.4f}"])
        coef_data.append(["Numeric time (normalized 0-1)", f"{model['numeric_time']:.4f}"])
        coef_data.append(["Max speed (km/h)", f"{model['max_speed']:.4f}"])
        coef_data.append(["School holidays (binary)", f"{model['school_holidays']:.4f}"])
        coef_data.append(["Illuminance (lux)", f"{model['illuminance']:.2e}"])
        
        for year, coef in model['source_year'].items():
            coef_data.append([f"Year {year}", f"{coef:.4f}"])
        
        for cat, coef in model['urban'].items():
            if coef != 0:
                coef_data.append([f"Urban - {cat}", f"{coef:.4f}"])
        
        for cat, coef in list(model['intersection'].items())[:5]:  # Top 5
            if coef != 0:
                coef_data.append([f"Intersection - {cat}", f"{coef:.4f}"])
        
        coef_df = pd.DataFrame(coef_data, columns=["Variable", "Coefficient"])
        st.dataframe(coef_df, use_container_width=True, hide_index=True)
        
        st.caption(f"📐 **Prediction formula**: Intercept + sum of all effects based on selected inputs")
        st.caption("⚠️ **Note**: Coefficients from GLM fitted on French accident database (2019-2024, n=320,488)")
    
    st.markdown("---")
    st.caption("📊 **Model insights based on French accident data (2019-2024):**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("⏰ **Later time** → more people involved")
        st.caption("🏙️ **Urban areas** → more people involved")
    with col2:
        st.caption("🔄 **Complex intersections** → more people involved")
        st.caption("⭕ **Roundabouts** → fewer people involved")
    with col3:
        st.caption("📅 **Weekends** → more people involved")
        st.caption("⚡ **Higher speed** → more people involved")

st.markdown("---")
st.caption("**Project**: Statistical Modeling of Traffic Accident Data on Large Road Networks")
st.caption("**Data source**: French national road accident database (2019-2024)")
st.caption("**Authors**: F. Cola, M. Filoramo, G. Genouville, V. Mariani | **Supervisor**: S. Panzeri")
