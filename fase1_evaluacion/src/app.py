from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Cargar el mejor modelo y scaler
print("Cargando modelo y scaler...")
modelo = joblib.load('../modelos/RandomForest_modelo.pkl')
scaler = joblib.load('../modelos/scaler.pkl')
print("✓ Modelo y scaler cargados exitosamente")

# Base de datos simulada (en memoria)
predicciones_db = []

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'API de Parkinson - Clasificación',
        'version': '1.0.0',
        'endpoints': {
            'health': '/api/health',
            'predict': '/api/predict (POST)',
            'history': '/api/history/<patient_id> (GET)',
            'stats': '/api/stats (GET)'
        }
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Verificar que la API esté funcionando"""
    return jsonify({
        'status': 'healthy',
        'model': 'RandomForest Classifier',
        'timestamp': datetime.now().isoformat(),
        'total_predicciones': len(predicciones_db)
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint para predecir si un paciente tiene Parkinson
    
    JSON esperado:
    {
        "patient_id": "PAC001",
        "features": {
            "MDVP:Fo(Hz)": 119.086,
            "MDVP:Fhi(Hz)": 157.302,
            ...
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({
                'success': False,
                'error': 'Datos inválidos. Se requiere el campo "features"'
            }), 400
        
        # Convertir features a DataFrame
        features_dict = data['features']
        features_df = pd.DataFrame([features_dict])
        
        # Columnas esperadas (22 features del dataset)
        columnas_esperadas = [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
            'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ',
            'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3',
            'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
            'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
        ]
        
        # Verificar que todas las columnas estén presentes
        for col in columnas_esperadas:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Reordenar columnas
        features_df = features_df[columnas_esperadas]
        
        # Escalar features
        features_scaled = scaler.transform(features_df)
        
        # Hacer predicción
        prediccion = modelo.predict(features_scaled)[0]
        probabilidades = modelo.predict_proba(features_scaled)[0]
        
        # Interpretar resultado
        label = 'Parkinson' if prediccion == 1 else 'Sano'
        confianza = round(float(max(probabilidades)) * 100, 2)
        
        # Crear resultado
        resultado = {
            'timestamp': datetime.now().isoformat(),
            'patient_id': data.get('patient_id', 'unknown'),
            'prediction': int(prediccion),
            'prediction_label': label,
            'confidence': confianza,
            'probabilities': {
                'sano': round(float(probabilidades[0]) * 100, 2),
                'parkinson': round(float(probabilidades[1]) * 100, 2)
            },
            'recomendacion': 'Se recomienda evaluación médica' if prediccion == 1 else 'Sin indicadores de riesgo'
        }
        
        # Guardar en "base de datos"
        predicciones_db.append(resultado)
        
        return jsonify({
            'success': True,
            'data': resultado
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/history/<patient_id>', methods=['GET'])
def get_history(patient_id):
    """Obtener historial de predicciones de un paciente"""
    history = [p for p in predicciones_db if p['patient_id'] == patient_id]
    
    return jsonify({
        'success': True,
        'patient_id': patient_id,
        'count': len(history),
        'data': history
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Estadísticas generales"""
    if len(predicciones_db) == 0:
        return jsonify({
            'success': True,
            'total_predicciones': 0,
            'message': 'Sin predicciones aún'
        })
    
    total = len(predicciones_db)
    parkinson = sum(1 for p in predicciones_db if p['prediction'] == 1)
    sano = total - parkinson
    
    return jsonify({
        'success': True,
        'total_predicciones': total,
        'parkinson': parkinson,
        'sano': sano,
        'porcentaje_parkinson': round((parkinson / total) * 100, 2)
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print(" INICIANDO API DE CLASIFICACIÓN PARKINSON")
    print("="*60)
    print(f"📍 URL: http://localhost:5000")
    print(f" Health Check: http://localhost:5000/api/health")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000)