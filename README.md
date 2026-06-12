# 🧠 Sistema de Detección de Parkinson con IA

Aplicación web para médicos que integra ML para monitoreo y gestión de pacientes con Parkinson.

## 📋 Descripción del Proyecto

Este sistema utiliza algoritmos de Machine Learning para analizar características de voz y predecir la presencia de Parkinson. La solución incluye:

- **Fase 1**: Evaluación comparativa de 3 algoritmos (RandomForest, SVM, KNN)
- **Fase 2**: Prototipo funcional con el algoritmo ganador (SVM) desplegado como API REST

## 🏆 Resultados de Evaluación (Fase 1)

| Algoritmo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-----------|----------|-----------|--------|----------|---------|
| Random Forest | 0.9231 | 0.9333 | 0.9655 | 0.9492 | 0.9621 |
| **SVM** ⭐ | **0.9231** | **0.9062** | **1.0** | **0.9508** | 0.9552 |
| KNN | 0.9231 | 0.9333 | 0.9655 | 0.9492 | 0.9638 |

**Ganador: SVM** seleccionado por mejor F1-Score y Recall perfecto.

## 🚀 API Endpoints (Fase 2)

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/health` | GET | Verificar estado de la API |
| `/api/predict` | POST | Predecir Parkinson |
| `/api/history/<id>` | GET | Historial de predicciones |
| `/api/stats` | GET | Estadísticas generales |
| `/api/model-info` | GET | Información del modelo |

## 📦 Instalación Local

```bash
cd fase2_prototipo
pip install -r requirements.txt
cd src
python algoritmo_svm.py    # Entrenar modelo
python api_svm.py          # Iniciar API
