import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

# Crear carpeta de modelos si no existe
os.makedirs('../modelos', exist_ok=True)

print("="*60)
print("ENTRENAMIENTO DE MODELOS - CLASIFICACIÓN PARKINSON")
print("="*60)

# 1. Cargar dataset
print("\n[1/5] Cargando dataset...")
df = pd.read_csv('../datos/parkinsons.data')
print(f"✓ Dataset cargado: {len(df)} registros")
print(f"✓ Columnas: {list(df.columns)}")

# 2. Preparar datos
print("\n[2/5] Preparando datos...")

# Eliminar columnas no necesarias (name es identificador, status es target)
X = df.drop(['name', 'status'], axis=1)
y = df['status']

print(f"✓ Features: {X.shape[1]} columnas")
print(f"✓ Target: status (0=Sano, 1=Parkinson)")
print(f"✓ Distribución: {y.value_counts().to_dict()}")

# Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalar features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardar scaler
joblib.dump(scaler, '../modelos/scaler.pkl')

print(f"✓ Datos de entrenamiento: {len(X_train)}")
print(f"✓ Datos de prueba: {len(X_test)}")

# 3. Definir los 3 algoritmos
print("\n[3/5] Definiendo algoritmos de clasificación...")
modelos = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}
print("✓ Algoritmos: Random Forest, SVM, KNN")

# 4. Entrenar y evaluar
print("\n[4/5] Entrenando y evaluando modelos...")
print("-"*60)

resultados = {}

for nombre, modelo in modelos.items():
    print(f"\n Entrenando {nombre}...")
    
    # Entrenar (todos con datos escalados para consistencia)
    modelo.fit(X_train_scaled, y_train)
    
    # Predecir
    y_pred = modelo.predict(X_test_scaled)
    y_pred_proba = modelo.predict_proba(X_test_scaled)[:, 1]
    
    # Calcular métricas
    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall': round(recall_score(y_test, y_pred), 4),
        'f1_score': round(f1_score(y_test, y_pred), 4),
        'auc_roc': round(roc_auc_score(y_test, y_pred_proba), 4)
    }
    
    resultados[nombre] = metrics
    
    # Guardar modelo
    ruta_modelo = f'../modelos/{nombre}_modelo.pkl'
    joblib.dump(modelo, ruta_modelo)
    
    print(f"✓ {nombre} entrenado y guardado")
    print(f"  Métricas: {metrics}")

# 5. Seleccionar mejor modelo (mejor F1-Score)
print("\n" + "="*60)
print("RESULTADOS FINALES")
print("="*60)

mejor_modelo = max(resultados, key=lambda x: resultados[x]['f1_score'])
print(f"\n MEJOR MODELO: {mejor_modelo}")
print(f"📊 Métricas del ganador:")
for metrica, valor in resultados[mejor_modelo].items():
    print(f"   {metrica}: {valor}")

# Guardar resultados
with open('../resultados_evaluacion.json', 'w') as f:
    json.dump({
        'dataset': 'Parkinsons (UCI) - parkinsons.data',
        'target': 'status (0=Sano, 1=Parkinson)',
        'total_registros': len(df),
        'resultados': resultados,
        'mejor_modelo': mejor_modelo
    }, f, indent=2)

print("\n✓ Resultados guardados en 'resultados_evaluacion.json'")
print("\n" + "="*60)
print("✅ PROCESO COMPLETADO")
print("="*60)