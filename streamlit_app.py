# app.py - Versione con auto-addestramento all'avvio
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import time
import plotly.express as px
import os
import sys

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
        .training-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Funzione per addestrare il modello
def train_model():
    """Addestra il modello e lo salva"""
    with st.spinner('🧠 Addestramento del modello in corso... Attendere qualche secondo'):
        try:
            # Import qui per evitare conflitti
            import numpy as np
            import pandas as pd
            from statsmodels.regression.mixed_linear_model import MixedLM
            from statsmodels.tools import add_constant
            
            # Simula dati di esempio
            np.random.seed(42)
            n = 500
            gruppi = np.repeat(range(50), 10)
            
            # Variabili
            ora = np.random.uniform(6, 22, n)
            temperatura = np.random.normal(20, 8, n)
            umidita = np.random.normal(65, 15, n)
            pioggia = np.random.binomial(1, 0.3, n)
            
            # Effetti casuali per gruppo
            effetti_casuali = np.random.normal(0, 2, 50)
            effetto_gruppo = effetti_casuali[gruppi]
            
            # Variabile target
            target = (
                5 + 
                0.3 * ora + 
                0.5 * temperatura - 
                0.2 * umidita - 
                1.5 * pioggia + 
                effetto_gruppo + 
                np.random.normal(0, 1, n)
            )
            
            df = pd.DataFrame({
                'target': target,
                'ora': ora,
                'temperatura': temperatura,
                'umidita': umidita,
                'pioggia': pioggia,
                'gruppo': gruppi
            })
            
            # Prepara per statsmodels
            X = add_constant(df[['ora', 'temperatura', 'umidita', 'pioggia']])
            y = df['target']
            groups = df['gruppo']
            
            # Modello misto
            model = MixedLM(y, X, groups)
            result = model.fit(disp=False)  # disp=False per silenziare output
            
            # Salva modello
            model_data = {
                'model': result,
                'feature_names': ['ora', 'temperatura', 'umidita', 'pioggia'],
                'coefficienti': result.params.to_dict()
            }
            
            with open('mixed_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            return True, result
        except Exception as e:
            return False, str(e)

# Carica o addestra modello
@st.cache_resource
def load_or_train_model():
    """Carica il modello se esiste, altrimenti lo addestra"""
    model_file = 'mixed_model.pkl'
    
    if not os.path.exists(model_file):
        # Mostra messaggio di training
        st.info("📊 **Primo avvio**: il modello sta venendo addestrato...")
        success, result = train_model()
        if success:
            st.success("✅ Modello addestrato con successo!")
            # Ricarica il modello
            with open(model_file, 'rb') as f:
                return pickle.load(f)
        else:
            st.error(f"❌ Errore nell'addestramento: {result}")
            st.stop()
    else:
        # Carica modello esistente
        with open(model_file, 'rb') as f:
            return pickle.load(f)

# Carica modello
try:
    model_data = load_or_train_model()
    model = model_data['model']
except Exception as e:
    st.error(f"❌ Errore critico: {e}")
    st.stop()

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

# Converti pioggia
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
    
    from statsmodels.tools import add_constant
    input_with_const = add_constant(input_data)
    
    try:
        prediction = model.predict(input_with_const, exog_re=None)[0]
        
        st.markdown(f"""
        <div class="prediction-card">
            <div style="font-size: 1.1rem;">📈 PREVISIONE</div>
            <div class="prediction-value">{prediction:.2f}</div>
            <div style="font-size: 0.9rem;">unità di misura</div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📋 **Dettagli input**"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("🕐 Ora", f"{orario.strftime('%H:%M')}")
                st.metric("💧 Umidità", f"{umidita}%")
            with col_b:
                st.metric("🌡️ Temperatura", f"{temperatura}°C")
                st.metric("🌧️ Condizioni", pioggia)
        
        with st.expander("🔬 **Info modello statistico**"):
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

else:
    st.info("👆 **Imposta i valori sopra e premi PREVEDI**")
    
    st.markdown("---")
    st.caption("📊 **Esempio di funzionamento** - Il modello tiene conto di:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🕐 **Effetto ora**\n(solitamente positivo)")
    with col2:
        st.caption("🌡️ **Effetto temperatura**\n(varia per contesto)")
    with col3:
        st.caption("📦 **Effetti casuali**\n(per gruppo/stazione)")

st.markdown("---")
st.caption("💡 **Nota**: Modello dimostrativo con dati simulati | Addestrato automaticamente")
st.caption("📱 App ottimizzata per dispositivi mobili | Mixed Linear Model")
