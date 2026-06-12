"""
API REST para consumir el algoritmo SVM de detección de Parkinson
Fase 2: Prototipo - API independiente
"""

from flask import Flask, request, jsonify
from algoritmo_svm import ParkinsonSVMClassifier
from datetime import datetime
import json
import os

app = Flask(__name__)

# Inicializar clasificador y cargar modelo
print("Inicializando API de Parkinson...")
clasificador = ParkinsonSVMClassifier()

# Cargar modelo pre-entrenado
modelo_path = '../modelos/svm_parkinson.pkl'
scaler_path = '../modelos/scaler.pkl'

if os.path.exists(modelo_path) and os.path.exists(scaler_path):
    clasificador.cargar_modelo(modelo_path, scaler_path)
    print("✓ Modelo SVM cargado exitosamente")
else:
    print("⚠ Modelo no encontrado. Entrenando nuevo modelo...")
    # Si no existe el modelo, entrenarlo
    X, y = clasificador.cargar_datos('../../datos/parkinsons.data')
    clasificador.entrenar(X, y)
    clasificador.guardar_modelo(modelo_path, scaler_path)

# Base de datos en memoria para almacenar predicciones
predicciones_db = []

@app.route('/', methods=['GET'])
def home():
    """Endpoint raíz con información de la API"""
    return jsonify({
        'message': 'API de Detección de Parkinson - SVM',
        'version': '2.0.0',
        'fase': 'Prototipo',
        'algoritmo': 'SVM (Support Vector Machine)',
        'endpoints': {
            'health': '/api/health (GET)',
            'predict': '/api/predict (POST)',
            'history': '/api/history/<patient_id> (GET)',
            'stats': '/api/stats (GET)',
            'model_info': '/api/model-info (GET)'
        }
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Verificar estado de la API"""
    return jsonify({
        'status': 'healthy',
        'model': 'SVM (Support Vector Machine)',
        'kernel': 'rbf',
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
        
        # Hacer predicción usando el algoritmo
        resultado_prediccion = clasificador.predecir(data['features'])
        
        # Crear registro completo
        resultado = {
            'timestamp': datetime.now().isoformat(),
            'patient_id': data.get('patient_id', 'unknown'),
            **resultado_prediccion,
            'recomendacion': 'Se recomienda evaluación médica' if resultado_prediccion['prediction'] == 1 else 'Sin indicadores de riesgo'
        }
        
        # Guardar en base de datos
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
    """Estadísticas generales de predicciones"""
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

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Información del modelo"""
    return jsonify({
        'success': True,
        'model': {
            'name': 'SVM (Support Vector Machine)',
            'kernel': 'rbf',
            'C': 1.0,
            'probability': True,
            'features_count': 22,
            'training_date': datetime.now().isoformat()
        },
        'performance': {
            'accuracy': 0.9231,
            'precision': 0.9062,
            'recall': 1.0,
            'f1_score': 0.9508,
            'auc_roc': 0.9552
        }
    })

if __name__ == '__main__':
    import os
    
    # Detectar si estamos en producción (Render) o desarrollo local
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print("\n" + "="*60)
    print("🚀 INICIANDO API DE PARKINSON - FASE 2 PROTOTIPO")
    print("="*60)
    print(f"📍 Puerto: {port}")
    print(f"📍 Modo debug: {debug}")
    print(f"📍 Health Check: http://localhost:{port}/api/health")
    print(f"📍 Predicción: http://localhost:{port}/api/predict (POST)")
    print("="*60 + "\n")
    
    # En producción usar host 0.0.0.0 para aceptar conexiones externas
    app.run(host='0.0.0.0', port=port, debug=debug)
