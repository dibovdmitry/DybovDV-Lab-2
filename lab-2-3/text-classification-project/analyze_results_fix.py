import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. Настройка URI ---
# URI должен совпадать с тем, что используется в mlflow_integration_fix.py
TRACKING_URI = "http://localhost:5000"
# ИСПРАВЛЕНИЕ: Используем корректное имя эксперимента с дефисами для соответствия mlflow_integration_fix.py
EXPERIMENT_NAME = "Emotion-Classification-FineTuning"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

print(f"Подключение к MLflow на URI: {TRACKING_URI}")
print(f"Поиск эксперимента: '{EXPERIMENT_NAME}'")

# --- 2. Поиск и Валидация Эксперимента ---
# ИСПРАВЛЕНИЕ: Добавлен set_tracking_uri, чтобы обеспечить подключение к серверу
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    print(f"\n❌ ОШИБКА: Эксперимент '{EXPERIMENT_NAME}' не найден.")
    print("Возможные причины:")
    print("1. MLflow Tracking Server (mlflow ui) не запущен по адресу http://localhost:5000.")
    print("2. В файле 'hyperparameter_tuning.py' еще не было запусков, которые бы создали этот эксперимент.")
    
    # Попытка создать эксперимент, чтобы избежать сбоя
    try:
        experiment_id = client.create_experiment(EXPERIMENT_NAME)
        experiment = client.get_experiment(experiment_id)
        print(f"Эксперимент '{EXPERIMENT_NAME}' был создан, но пока пуст.")
    except Exception as e:
        print(f"Не удалось создать эксперимент: {e}")
        # Завершаем скрипт, если не удалось ни найти, ни создать
        exit(1)

# --- 3. Анализ Запусков ---
# Используем experiment.experiment_id, который теперь гарантированно не None
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    # ИСПРАВЛЕНИЕ: Сортируем по Best Validation F1, т.к. это основная метрика
    order_by=["metrics.best_validation_f1 DESC"] 
)

if not runs:
    print("\n✅ Эксперимент найден, но не содержит ни одного запуска. Запустите 'hyperparameter_tuning.py' сначала.")
    exit(0)

# --- 4. Вывод Результатов ---
print(f"\nНайдено {len(runs)} запусков.")
results = []

for run in runs:
    metrics = {k: v for k, v in run.data.metrics.items() if not k.startswith("test_")}
    
    results.append({
        "Run ID": run.info.run_id,
        "LR": run.data.params.get("learning_rate", "N/A"),
        "Best Val F1": run.data.metrics.get("best_validation_f1", 0.0),
        "Test F1": run.data.metrics.get("test_eval_f1_score", run.data.metrics.get("test_f1_score", 0.0)),
        "Checkpoint": run.data.tags.get("best_checkpoint_path", "N/A")
    })

df = pd.DataFrame(results)

print("\n--- Сводка Запусков (Отсортировано по Best Val F1) ---")
print(df.sort_values(by="Best Val F1", ascending=False).to_markdown(index=False))

best_run = df.sort_values(by="Best Val F1", ascending=False).iloc[0]
print("\n--- Лучший Результат ---")
pprint(best_run.to_dict())
