import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

def get_data():
    """Carga y devuelve el dataset de reviews de Bebes (Baby_5.json)."""
    # Buscamos el archivo en el directorio actual
    path = os.path.join(os.path.dirname(__file__), 'Baby_5.json')
    
    # Cargamos 10000 filas para que sea manejable pero representativo
    df = pd.read_json(path, lines=True, chunksize=10000)
    df = next(df)
    
    # Seleccionamos columnas relevantes y limpiamos
    df = df[['reviewText', 'overall']].copy()
    df = df.dropna(subset=['reviewText'])
    
    # 4 o 5 estrellas = Positivo (1), el resto = Negativo (0)
    df['target'] = (df['overall'] > 3).astype(int)
    
    return df

def preprocess_and_split(df):
    """Divide los datos en entrenamiento y test."""
    X = df['reviewText']
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_pipeline(n_estimators=100, max_features=1000):
    """Crea un Pipeline con TfidfVectorizer y RandomForestClassifier."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=max_features, stop_words='english')),
        ('rf', RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight="balanced"))
    ])

def evaluate_model(model, X_test, y_test):
    """Calcula y devuelve las métricas de evaluación."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    }
    return metrics
