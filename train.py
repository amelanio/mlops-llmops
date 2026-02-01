import argparse
import mlflow
import mlflow.sklearn
from model_utils import get_data, preprocess_and_split, build_pipeline, evaluate_model

def run_training(n_estimators, max_features):
    # Configuración de MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("practica-final-nlp")

    # 1. Obtener datos
    print("Cargando datos de Baby_5.json...")
    df = get_data()
    X_train, X_test, y_train, y_test = preprocess_and_split(df)

    # 2. Iniciar Run de MLflow
    with mlflow.start_run(run_name=f"NLP_RF_estimators_{n_estimators}_features_{max_features}"):
        # 3. Construir y entrenar Pipeline
        print(f"Entrenando modelo con {n_estimators} estimadores y {max_features} características...")
        pipeline = build_pipeline(n_estimators=n_estimators, max_features=max_features)
        pipeline.fit(X_train, y_train)

        # 4. Evaluar
        metrics = evaluate_model(pipeline, X_test, y_test)

        # 5. Loggear parámetros y métricas
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_features", max_features)
        
        for name, val in metrics.items():
            mlflow.log_metric(name, val)

        # 6. Loggear modelo
        # Registramos como 'baby-sentiment-model'
        mlflow.sklearn.log_model(pipeline, "model", registered_model_name="baby-sentiment-model")
        
        print(f"Entrenamiento completado. Métricas: {metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script de entrenamiento NLP para la práctica final.')
    parser.add_argument('--n_estimators', type=int, default=300, help='Número de árboles en el bosque.')
    parser.add_argument('--max_features', type=int, default=2000, help='Número máximo de características para TF-IDF.')
    
    args = parser.parse_args()
    
    run_training(n_estimators=args.n_estimators, max_features=args.max_features)
