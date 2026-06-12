"""
Algoritmo de Machine Learning - SVM para detección de Parkinson
Fase 2: Prototipo con algoritmo ganador
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import json
import os

class ParkinsonSVMClassifier:
    """
    Clasificador SVM para detección de Parkinson basado en características de voz
    """
    
    def __init__(self):
        self.modelo = None
        self.scaler = None
        self.metricas = None
        
    def cargar_datos(self, ruta_datos):
        """Cargar y preparar el dataset"""
        print(f"Cargando dataset desde: {ruta_datos}")
        df = pd.read_csv(ruta_datos)
        
        # Separar features y target
        X = df.drop(['name', 'status'], axis=1)
        y = df['status']
        
        print(f"✓ Dataset cargado: {len(df)} registros")
        print(f"✓ Features: {X.shape[1]} columnas")
        print(f"✓ Distribución de clases: {y.value_counts().to_dict()}")
        
        return X, y
    
    def entrenar(self, X, y, test_size=0.2):
        """Entrenar el modelo SVM"""
        print("\n" + "="*60)
        print("ENTRENANDO MODELO SVM - FASE 2 PROTOTIPO")
        print("="*60)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Escalar features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nDatos de entrenamiento: {len(X_train)}")
        print(f"Datos de prueba: {len(X_test)}")
        
        # Crear y entrenar modelo SVM
        print("\nEntrenando SVM...")
        self.modelo = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
        self.modelo.fit(X_train_scaled, y_train)
        
        # Evaluar
        y_pred = self.modelo.predict(X_test_scaled)
        y_pred_proba = self.modelo.predict_proba(X_test_scaled)[:, 1]
        
        # Calcular métricas
        self.metricas = {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred), 4),
            'recall': round(recall_score(y_test, y_pred), 4),
            'f1_score': round(f1_score(y_test, y_pred), 4),
            'auc_roc': round(roc_auc_score(y_test, y_pred_proba), 4)
        }
        
        print(f"\n✓ Modelo entrenado exitosamente")
        print(f"📊 Métricas del modelo:")
        for metrica, valor in self.metricas.items():
            print(f"   {metrica}: {valor}")
        
        return self.metricas
    
    def predecir(self, features_dict):
        """
        Hacer predicción con nuevos datos
        
        Args:
            features_dict: Diccionario con las 22 features del paciente
        
        Returns:
            Diccionario con resultado de la predicción
        """
        if self.modelo is None or self.scaler is None:
            raise Exception("El modelo no ha sido entrenado aún")
        
        # Convertir a DataFrame
        features_df = pd.DataFrame([features_dict])
        
        # Columnas esperadas
        columnas_esperadas = [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
            'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ',
            'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3',
            'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
            'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
        ]
        
        # Verificar columnas
        for col in columnas_esperadas:
            if col not in features_df.columns:
                features_df[col] = 0
        
        features_df = features_df[columnas_esperadas]
        
        # Escalar y predecir
        features_scaled = self.scaler.transform(features_df)
        prediccion = self.modelo.predict(features_scaled)[0]
        probabilidades = self.modelo.predict_proba(features_scaled)[0]
        
        return {
            'prediction': int(prediccion),
            'prediction_label': 'Parkinson' if prediccion == 1 else 'Sano',
            'confidence': round(float(max(probabilidades)) * 100, 2),
            'probabilities': {
                'sano': round(float(probabilidades[0]) * 100, 2),
                'parkinson': round(float(probabilidades[1]) * 100, 2)
            }
        }
    
    def guardar_modelo(self, ruta_modelo, ruta_scaler):
        """Guardar modelo y scaler"""
        joblib.dump(self.modelo, ruta_modelo)
        joblib.dump(self.scaler, ruta_scaler)
        print(f"\n✓ Modelo guardado en: {ruta_modelo}")
        print(f"✓ Scaler guardado en: {ruta_scaler}")
    
    def cargar_modelo(self, ruta_modelo, ruta_scaler):
        """Cargar modelo y scaler existentes"""
        self.modelo = joblib.load(ruta_modelo)
        self.scaler = joblib.load(ruta_scaler)
        print(f"✓ Modelo cargado desde: {ruta_modelo}")
        print(f"✓ Scaler cargado desde: {ruta_scaler}")


# Script principal para entrenar y guardar el modelo
if __name__ == '__main__':
    # Crear clasificador
    clasificador = ParkinsonSVMClassifier()
    
    # Cargar datos
    X, y = clasificador.cargar_datos('../../datos/parkinsons.data')
    
    # Entrenar
    metricas = clasificador.entrenar(X, y)
    
    # Guardar modelo
    os.makedirs('../modelos', exist_ok=True)
    clasificador.guardar_modelo(
        '../modelos/svm_parkinson.pkl',
        '../modelos/scaler.pkl'
    )
    
    # Guardar métricas
    with open('../metricas_modelo.json', 'w') as f:
        json.dump({
            'algoritmo': 'SVM (Support Vector Machine)',
            'kernel': 'rbf',
            'metricas': metricas
        }, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ ALGORITMO SVM ENTRENADO Y GUARDADO")
    print("="*60)
    
    # Prueba de predicción
    print("\n🧪 Prueba de predicción con datos de ejemplo:")
    datos_prueba = {
        "MDVP:Fo(Hz)": 119.086,
        "MDVP:Fhi(Hz)": 157.302,
        "MDVP:Flo(Hz)": 74.997,
        "MDVP:Jitter(%)": 0.00784,
        "MDVP:Jitter(Abs)": 0.00007,
        "MDVP:RAP": 0.0037,
        "MDVP:PPQ": 0.00554,
        "Jitter:DDP": 0.01109,
        "MDVP:Shimmer": 0.04374,
        "MDVP:Shimmer(dB)": 0.426,
        "Shimmer:APQ3": 0.02971,
        "Shimmer:APQ5": 0.03858,
        "MDVP:APQ": 0.04955,
        "Shimmer:DDA": 0.06545,
        "NHR": 0.02144,
        "HNR": 21.033,
        "RPDE": 0.414783,
        "DFA": 0.815285,
        "spread1": -4.813031,
        "spread2": 0.266482,
        "D2": 2.301442,
        "PPE": 0.284654
    }
    
    resultado = clasificador.predecir(datos_prueba)
    print(f"Resultado: {resultado}")