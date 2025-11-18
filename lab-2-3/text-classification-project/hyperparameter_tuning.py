import mlflow
from mlflow_integration_fix3 import train_model

with mlflow.start_run(run_name="Hyperparameter_Search"):
    mlflow.set_tag("experiment_type", "learning_rate_tuning") 

    learning_rates = [1e-5, 2e-5, 5e-5]

    for lr in learning_rates:
        with mlflow.start_run(nested=True, run_name=f"LR={lr}"):
            mlflow.log_param("learning_rate", lr)
            results = train_model(learning_rate=lr)             
            mlflow.log_metrics(results)

print("Эксперимент по подбору learning rate завершен и результаты сохранены в MLflow!")
