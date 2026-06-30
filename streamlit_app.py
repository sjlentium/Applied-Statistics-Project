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
    initial_sidebar_state="expanded" # Set back to expanded by default
)

# Keep all project information in the sidebar as requested
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

# Mobile-friendly CSS with Media Queries
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
    """
    try:
        point_fixed_effects = pd.read_csv(DATA_DIR / "point_fixed_effects.csv").set_index("term")["estimate"]
        bootstrap_fixed_effects = pd.read_csv(DATA_DIR / "bootstrap_fixed_effects.csv.gz").drop(columns="bootstrap_id")
        point_random_effects =
        
