import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # forzar CPU, evitar líos con la GPU

import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# ---- 1. Cargar modelo (solo para predecir, sin recompilar) ----
model = load_model("modelo_calidad_agua.h5", compile=False)

# ---- 2. Cargar scaler + nombres de variables ----
with open("scaler_features.pkl", "rb") as f:
    data = pickle.load(f)

scaler = data["scaler"]
feature_names = data["features"]

print("Variables usadas por el modelo:")
print(feature_names)

# ---- 3. Leer dataset original y repetir el mismo preprocesamiento ----
df = pd.read_csv("BBDD_FINAL_sintetica.csv")

# quitamos las columnas que se eliminaron en el entrenamiento
cols_drop = ["station", "year", "season"]
df = df.drop(columns=[c for c in cols_drop if c in df.columns])

# también quitamos la variable objetivo (por ejemplo "OD")
target_col = "OD"  # cámbiala si tu y fue otra
df_features = df.drop(columns=[target_col])

# nos aseguramos de usar SOLO las columnas que vio el modelo, en el mismo orden
df_features = df_features[feature_names]

# ---- 4. Tomar una fila de ejemplo ----
fila_df = df_features.iloc[[0]]  # doble corchete para que siga siendo DataFrame
print("\nFila usada para ejemplo (features):")
print(fila_df)

# ---- 5. Escalar y predecir ----
X = scaler.transform(fila_df)
pred = model.predict(X)

print("\nPredicción para esa fila:")
print(pred[0][0])
