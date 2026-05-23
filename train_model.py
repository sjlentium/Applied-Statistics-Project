# train_model.py - Esegui questo PRIMA di app.py
import numpy as np
import pandas as pd
import pickle
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools import add_constant

print("🔄 Generazione dati simulati...")

# Simula dati di esempio
np.random.seed(42)
n = 500
gruppi = np.repeat(range(50), 10)  # 50 gruppi (es. stazioni meteo)

# Variabili
ora = np.random.uniform(6, 22, n)  # ore del giorno
temperatura = np.random.normal(20, 8, n)
umidita = np.random.normal(65, 15, n)
pioggia = np.random.binomial(1, 0.3, n)

# Effetti casuali per gruppo
effetti_casuali = np.random.normal(0, 2, 50)
effetto_gruppo = effetti_casuali[gruppi]

# Variabile target (es. consumo energetico, produzione agricola, etc)
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

print("📊 Dataset creato con 500 osservazioni")

# Prepara per statsmodels
X = add_constant(df[['ora', 'temperatura', 'umidita', 'pioggia']])
y = df['target']
groups = df['gruppo']

print("🧠 Addestramento Mixed Model...")

# Modello misto con intercetta casuale per gruppo
model = MixedLM(y, X, groups)
result = model.fit()

print("\n✅ Modello addestrato!")
print(result.summary())

# Salva modello e scaler (per preprocessing)
model_data = {
    'model': result,
    'feature_names': ['ora', 'temperatura', 'umidita', 'pioggia'],
    'coefficienti': result.params.to_dict()
}

with open('mixed_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\n💾 Modello salvato come 'mixed_model.pkl'")
print("\n📌 Ora puoi eseguire: streamlit run app.py")
