# streamlit_app.py - Accident Severity Predictor using GLM Model
# Authors: F. Cola, M. Filoramo, G. Genouville, V. Mariani
# Tutor: Simone Panzeri
# Project: Statistical Modeling of Traffic Accident Data on Large Road Networks

import streamlit as st
import numpy as np
from datetime import time
import pandas as pd

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
    
    Using a Generalized Linear Model (GLM) fitted on the French national road accident database, this tool calculates the logistic probability based on:
    - **Temporal factors** (cyclic time, month, day type)
    - **Road infrastructure** (intersection type, traffic regime, turn, surface)
    - **Environmental conditions** (urban/rural setting, weather, brightness)
    """)

# Mobile-friendly CSS
st.markdown("""
    <style>
        .stButton button { width: 100%; padding: 0.75rem; font-size: 1.2rem; background-color: #4CAF50; color: white; border-radius: 12px; }
        .stSelectbox, .stSlider, .stTimeInput { margin-bottom: 1rem; }
        .prediction-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 20px; color: white; text-align: center; margin: 1rem 0; }
        .prediction-value { font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """
    Load the pre-trained baseline GLM model coefficients.
    Note: The exact 59 parameters are missing from the text. 
    They have been instantiated with 0.0 placeholders and must be updated.
    """
    return {
        'intercept': 0.0,
        
        # Categorical variables (Reference levels inherently equal 0.0)
        'day_type': {
            'Work day': 0.0, # Reference level
            'Saturday': 0.0,
            'Public holiday or Sunday': 0.0
        },
        'intersection': {
            'No intersection': 0.0, # Reference level
            'X-shaped': 0.0,
            'T-shaped': 0.0,
            'Y-shaped': 0.0,
            '4 or more branches': 0.0,
            'Roundabout': 0.0,
            'Square': 0.0,
            'Other intersection': 0.0
        },
        'traffic': {
            'Two-way': 0.0, # Reference level
            'One-way': 0.0,
            'Dual carriageway': 0.0,
            'Dynamical lane assignment': 0.0,
            'Unknown': 0.0
        },
        'surface': {
            'Normal': 0.0, # Reference level
            'Wet': 0.0,
            'Snow or ice': 0.0,
            'Other': 0.0
        },
        'turn': {
            'Straight': 0.0, # Reference level
            'Left bend': 0.0,
            'Right bend': 0.0,
            'S-shaped bend': 0.0
        },
        'urban': {
            'Urban': 0.0, # Reference level
            'Rural': 0.0
        },
        'weather': {
            'Normal': 0.0, # Reference level
            'Drizzle': 0.0,
            'Snow or hail': 0.0,
            'Smoke or fog': 0.0,
            'Dazzling': 0.0,
            'Overcast': 0.0,
            'Storm or heavy rain': 0.0,
            'Other': 0.0
        },
        'brightness': {
            'Broad daylight': 0.0, # Reference level
            'Dusk or dawn': 0.0,
            'Night with street lighting on': 0.0,
            'Night without street lighting': 0.0
        },
        
        # Continuous variables (Standardized during modeling, requires training means/stdevs)
        'max_speed': 0.0, 
        'precipitation': 0.0,
        'relative_humidity': 0.0,
        
        # Cyclic time components
        'time_sin': 0.0,
        'time_cos': 0.0
    }

model = load_model()

st.title("🚗 **Serious-Injury Risk Predictor**")
st.markdown("*Baseline GLM prediction of injury probability (Metropolitan France & Corsica)*")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📅 **Temporal & Environmental**")
    
    # Cyclic time encoding as described in the paper
    accident_time = st.time_input("Accident time", value=time(12, 0))
    numeric_hour = accident_time.hour + accident_time.minute / 60.0
    
    # Compute sine and cosine transformations for 24-hour cycle
    time_sin = np.sin((2 * np.pi * numeric_hour) / 24)
    time_cos = np.cos((2 * np.pi * numeric_hour) / 24)
    
    day_type = st.selectbox("Day type", ["Work day", "Saturday", "Public holiday or Sunday"])
    weather = st.selectbox("Weather", ["Normal", "Drizzle", "Snow or hail", "Smoke or fog", "Dazzling", "Overcast", "Storm or heavy rain", "Other"])
    brightness = st.selectbox("Brightness", ["Broad daylight", "Dusk or dawn", "Night with street lighting on", "Night without street lighting"])
    
    precipitation = st.slider("Total precipitation (mm)", 0.0, 100.0, 0.0)
    relative_humidity = st.slider("Relative humidity (%)", 0.0, 100.0, 50.0)

with col2:
    st.subheader("🏗️ **Road Infrastructure**")
    
    urban = st.selectbox("Area type", ["Urban", "Rural"])
    intersection = st.selectbox("Intersection", ["No intersection", "X-shaped", "T-shaped", "Y-shaped", "4 or more branches", "Roundabout", "Square", "Other intersection"])
    traffic = st.selectbox("Traffic pattern", ["Two-way", "One-way", "Dual carriageway", "Dynamical lane assignment", "Unknown"])
    surface = st.selectbox("Surface condition", ["Normal", "Wet", "Snow or ice", "Other"])
    turn = st.selectbox("Road layout (Turn)", ["Straight", "Left bend", "Right bend", "S-shaped bend"])
    
    max_speed = st.slider("Speed limit (km/h)", 20, 130, 50, step=10)

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🔮 **PREDICT PROBABILITY**")

if predict_button:
    # 1. Compute linear predictor (eta) by summing the log-odds coefficients
    eta = model['intercept']
    eta += model['day_type'].get(day_type, 0.0)
    eta += model['weather'].get(weather, 0.0)
    eta += model['brightness'].get(brightness, 0.0)
    eta += model['urban'].get(urban, 0.0)
    eta += model['intersection'].get(intersection, 0.0)
    eta += model['traffic'].get(traffic, 0.0)
    eta += model['surface'].get(surface, 0.0)
    eta += model['turn'].get(turn, 0.0)
    
    # In a fully deployed model, continuous variables must be standardized: (value - mean) / sd
    eta += model['max_speed'] * max_speed 
    eta += model['precipitation'] * precipitation
    eta += model['relative_humidity'] * relative_humidity
    eta += model['time_sin'] * time_sin
    eta += model['time_cos'] * time_cos
    
    # 2. Apply Logistic Link Function to calculate probability
    probability = 1 / (1 + np.exp(-eta))
    probability_percentage = probability * 100
    
    st.markdown(f"""
    <div class="prediction-card">
        <div style="font-size: 1.1rem;">PROBABILITY OF SERIOUS INJURY OR FATALITY</div>
        <div class="prediction-value">{probability_percentage:.2f}%</div>
        <div style="font-size: 0.9rem;">Given that a person is involved in the accident</div>
    </div>
    """, unsafe_allow_html=True)
