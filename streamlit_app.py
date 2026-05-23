# streamlit_app.py - App principale
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import time
import plotly.express as px

# Configurazione pagina (deve essere il primo comando Streamlit)
st.set_page_config(
    page_title="Mixed Model Predictor",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS per mobile
st.markdown("""
    <style>
        /* Miglioramenti per touch screen */
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
        /* Card effetto per risultati */
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

# Carica modello
@st.cache_resource
def load_model():
    try:
        with open('mixed_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("❌ Modello non trovato! Esegui prima 'python train_model.py'")
        st.stop()
    except Exception as e:
        st.error(f"❌ Errore nel caricamento: {e}")
        st.stop()

model_data = load_model()
model = model_data['model']

# Titolo
st.title("📊 **Predittore con Mixed Model**")
st.markdown("*Modello lineare a effetti misti per previsioni personalizzate*")
st.divider()

# Layout a colonne per input
col1, col2 = st.columns(2)

with col1:
    st.subheader("🕐 **Tempo**")
    orario = st.time_input(
        "Seleziona orario",
        value=time(12, 0),
        help="Scegli l'ora del giorno"
    )
    ora_decimale = orario.hour + orario.minute/60
    
    st.subheader("🌡️ **Temperatura**")
    temperatura = st.slider(
        "°C",
        min_value=-10,
        max_value=45,
        value=20,
        step=1,
        help="Temperatura in gradi Celsius"
    )

with col2:
    st.subheader("💧 **Umidità**")
    umidita = st.slider(
        "%",
        min_value=0,
        max_value=100,
        value=65,
        step=5,
        help="Umidità relativa"
    )
    
    st.subheader("🌧️ **Precipitazioni**")
    pioggia = st.selectbox(
        "Condizioni",
        options=["☀️ Sole", "☁️ Nuvoloso", "🌧️ Pioggia", "⛈️ Temporale"],
        help="Scegli le condizioni meteorologiche"
    )

# Converti pioggia in variabile numerica
pioggia_map = {
    "☀️ Sole": 0,
    "☁️ Nuvoloso": 0,
    "🌧️ Pioggia": 1,
    "⛈️ Temporale": 1
}
pioggia_val = pioggia_map[pioggia]

# Pulsante predizione
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🔮 **PREVEDI**", use_container_width=True)

# Visualizza risultato
if predict_button:
    # Crea dataframe per predizione
    input_data = pd.DataFrame({
        'ora': [ora_decimale],
        'temperatura': [temperatura],
        'umidita': [umidita],
        'pioggia': [pioggia_val]
    })
    
    # Aggiungi costante (intercetta) come nel training
    from statsmodels.tools import add_constant
    input_with_const = add_constant(input_data)
    
    # Predizione (senza effetti casuali - fixed effects only)
    try:
        prediction = model.predict(input_with_const, exog_re=None)[0]
        
        # Mostra risultato
        st.markdown("""
        <div class="prediction-card">
            <div style="font-size: 1.1rem;">📈 PREVISIONE</div>
            <div class="prediction-value">{:.2f}</div>
            <div style="font-size: 0.9rem;">unità di misura</div>
        </div>
        """.format(prediction), unsafe_allow_html=True)
        
        # Dettagli degli input
        with st.expander("📋 **Dettagli input**"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("🕐 Ora", f"{orario.strftime('%H:%M')}")
                st.metric("💧 Umidità", f"{umidita}%")
            with col_b:
                st.metric("🌡️ Temperatura", f"{temperatura}°C")
                st.metric("🌧️ Condizioni", pioggia)
        
        # Mostra coefficienti del modello
        with st.expander("🔬 **Info modello statistico**"):
            st.caption("Coefficienti del modello (effetti fissi):")
            coef_df = pd.DataFrame({
                'Variabile': ['Intercetta', 'Ora', 'Temperatura', 'Umidità', 'Pioggia'],
                'Coefficiente': [
                    model.params.get('const', 0),
                    model.params.get('ora', 0),
                    model.params.get('temperatura', 0),
                    model.params.get('umidita', 0),
                    model.params.get('pioggia', 0)
                ]
            })
            st.dataframe(coef_df, use_container_width=True, hide_index=True)
            
    except Exception as e:
        st.error(f"Errore nella predizione: {e}")
        st.info("Assicurati che il modello sia stato addestrato correttamente")

else:
    # Messaggio iniziale
    st.info("👆 **Imposta i valori sopra e premi PREVEDI**")
    
    # Grafico esplicativo
    st.markdown("---")
    st.caption("📊 **Esempio di funzionamento** - Il modello tiene conto di:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🕐 **Effetto ora**\n(solitamente positivo)")
    with col2:
        st.caption("🌡️ **Effetto temperatura**\n(varia per contesto)")
    with col3:
        st.caption("📦 **Effetti casuali**\n(per gruppo/stazione)")

# Footer
st.markdown("---")
st.caption("💡 **Nota**: Questo è un modello dimostrativo con dati simulati")
st.caption("📱 App ottimizzata per dispositivi mobili | Mixed Linear Model")
