from fastapi import FastAPI, HTTPException, Query
from transformers import pipeline
import mlflow
import mlflow.pyfunc
import os
import pandas as pd

app = FastAPI(title="API Práctica Final NLP - Bebés", description="API con integración de MLflow (Sentiment Analysis) y Hugging Face")

print("Cargando pipelines de Hugging Face...")
sentiment_hf_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
generator_pipeline = pipeline("text-generation", model="gpt2")

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_NAME = "baby-sentiment-model"

mlflow_model = None

def load_mlflow_model():
    global mlflow_model
    try:
        model_uri = f"models:/{MODEL_NAME}/latest"
        try:
            mlflow_model = mlflow.pyfunc.load_model(model_uri)
        except:
            mlflow_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/1")
        print(f"Modelo {MODEL_NAME} cargado correctamente desde MLflow.")
    except Exception as e:
        print(f"Error al cargar el modelo de MLflow: {e}")
        mlflow_model = None

@app.on_event("startup")
def startup_event():
    load_mlflow_model()

@app.get("/")
def read_root():
    """Endpoint de bienvenida."""
    return {
        "message": "Bienvenido a la API de la Práctica Final (NLP Bebés)",
        "endpoints_disponibles": ["/", "/status", "/model-info", "/predict", "/nlp/sentiment", "/nlp/generate"],
        "docs": "/docs"
    }

@app.get("/status")
def get_status():
    """Endpoint para comprobar el estado de la API."""
    return {"status": "ok", "version": "2.1.0", "mlflow_model_loaded": mlflow_model is not None}

@app.get("/model-info")
def get_model_info():
    """Endpoint que devuelve información sobre el modelo registrado en MLflow."""
    try:
        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions(MODEL_NAME)
        
        if not latest_versions:
            return {"message": f"No se han encontrado versiones para el modelo '{MODEL_NAME}'"}
            
        latest = latest_versions[-1]
        return {
            "model_name": MODEL_NAME,
            "version": latest.version,
            "stage": latest.current_stage,
            "run_id": latest.run_id,
            "creation_timestamp": latest.creation_timestamp
        }
    except Exception as e:
        return {"error": str(e), "message": "Asegúrate de que el servidor MLflow esté corriendo y el modelo esté registrado."}

@mlflow.trace
def predict_sentiment_trace(text: str) -> dict:
    """
    Función de predicción trazada en MLflow que reutiliza el modelo cargado.
    """
    if mlflow_model is None:
        load_mlflow_model()
        if mlflow_model is None:
            raise HTTPException(status_code=503, detail="Modelo de MLflow no disponible. ¿Has ejecutado el entrenamiento?")

    prediction = mlflow_model.predict([text])
    sentiment = "Positivo" if prediction[0] == 1 else "Negativo"
    return {
        "input_text": text,
        "prediction": sentiment,
        "class": int(prediction[0])
    }


@app.get("/predict")
def predict_sentiment(text: str = Query(..., description="Texto de la review para analizar sentimiento")):
    """
    Endpoint que utiliza el modelo entrenado (RandomForest + TF-IDF) cargado desde MLflow.
    """
    try:
        return predict_sentiment_trace(text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

@app.get("/nlp/sentiment")
def analyze_sentiment_hf(text: str):
    """
    Endpoint que utiliza un pipeline de Hugging Face para análisis de sentimiento.
    """
    if not text:
        raise HTTPException(status_code=400, detail="Debes proporcionar un texto")
    
    result = sentiment_hf_pipeline(text)
    return {"input_text": text, "hf_analysis": result}

@app.get("/nlp/generate")
def generate_text(text: str):
    """
    Endpoint que utiliza un pipeline de Hugging Face para generación de texto (GPT-2).
    """
    if not text:
        raise HTTPException(status_code=400, detail="Debes proporcionar un texto base")
    
    # Generamos una continuación del texto
    result = generator_pipeline(text, max_length=50, num_return_sequences=1)
    return {"input_text": text, "generated_text": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
