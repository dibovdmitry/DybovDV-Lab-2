# МИНИCTEPCTBO НАУКИ И ВЫСШЕГО ОБРАЗОВАНИЯ РОССИЙСКОЙ ФЕДЕРАЦИИ
## Федеральное государственное автономное образовательное учреждение высшего образования «Северо-Кавказский федеральный университет» Институт перспективной инженерии

### Отчет по лабораторной работе 2
### Знакомство с платформой Hugging Face Hub, тонкая настройка модели для текстовой классификации, интеграция MLflow для трекинга экспериментов
Дата: 2025-11-17 \
Семестр: [2 курс 1 полугодие - 3 семестр] \
Группа: ПИН-м-о-24-1 \
Дисциплина: Технологии программирования \
Студент: Дыбов Д.В.
#### Цель работы
Освоение базовых принципов работы с Hugging Face Hub; практическая тонкая настройка предобученных моделей для текстовой классификации; интеграция MLflow для трекинга экспериментов, логирования гиперпараметров/метрик/артефактов и моделей.
#### Теоретическая часть
Краткие изученные концепции:
- Hugging Face Hub: репозитории моделей, токенизаторы, датасеты.
- Trainer API: TrainingArguments, evaluation_strategy (evaluation_during_training при необходимости), compute_metrics.
- MLflow: настройка tracking_uri, mlflow.start_run, mlflow.log_param, mlflow.log_metric, mlflow.log_artifact / mlflow.log_artifacts, mlflow.log_model.
- Подготовка данных: train/validation split, токенизация, формат datasets.Dataset.
#### Практическая часть
##### Выполненные задачи
- [x] Установить пакеты: huggingface_hub, datasets, transformers, pandas, numpy.
- [x] Найти подходящую модель на Hugging Face Hub и изучить карточку модели.
- [x] Создать скрипт hf_hub_exploration.py — скачать модель/датасет и проверить содержимое.
- [x] Подготовить проект и файл resources.txt с информацией о выбранных ресурсах.
- [x] Создать fine_tuning.py (использован параметр evaluation_during_training при необходимости).
- [x] Запустить MLflow Tracking Server и интегрировать обучение через mlflow_integration.py.
- [x] Провести гиперпараметрическое исследование с hyperparameter_tuning.py.
- [x] Проанализировать результаты в analyze_results.py.
##### Ключевые фрагменты кода
- Скрипт hf_hub_exploration.py:
```python
# Пример интеграции тренировки с MLflow (схематично)
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Emotion-Classification-FineTuning")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)

training_args = TrainingArguments(
    output_dir="./outputs",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=500,
)

with mlflow.start_run():
    mlflow.log_param("model_name", "bert-base-uncased")
    mlflow.log_param("epochs", training_args.num_train_epochs)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, compute_metrics=compute_metrics)
    trainer.train()
    metrics = trainer.evaluate()
    mlflow.log_metrics(metrics)
    trainer.save_model("saved_model")
    mlflow.log_artifacts("saved_model")
```
##### Результаты выполнения
1. Установлены пакеты huggingface_hub, datasets, transformers, pandas, numpy. \ 
![скриншот](report/Screenshot1.png "Рисунок") \
Рисунок 1 – Установка пакетов \

2. На Hugging Face Hub найден подходящий репозиторий модели и изучена модельная карточка. \
![скриншот](report/Screenshot2.png "Рисунок") \
Рисунок 2 – Найденная модель на Hugging Face Hub \

3. Создан hf_hub_exploration.py, выполнено скачивание модели/датасета и проверка содержимого. \
![скриншот](report/Screenshot4.png "Рисунок") \
Рисунок 3 – Выполнение hf_hub_exploration.py \

4. Создана папка проекта и resources.txt с записями выбранных ресурсов. \
![скриншот](report/Screenshot5.png "Рисунок") \
Рисунок 4 – Создание проекта и resources.txt \

![скриншот](report/Screenshot6.png "Рисунок") \
Рисунок 5 – Проверка содержимого resources.txt \

5. Создан fine_tuning.py, в коде использован параметр eval_training вместо evaluation_training, запущен тренинг. \
![скриншот](report/Screenshot8-1.png "Рисунок") \
![скриншот](report/Screenshot8-2.png "Рисунок") \
Рисунок 6 – Запуск fine_tuning.py \

7. Запущен MLflow Tracking Server, произведена интеграция через mlflow_integration.py. \
![скриншот](report/Screenshot9.png "Рисунок") \
Рисунок 7 – MLflow Tracking Server \

![скриншот](report/Screenshot11.png "Рисунок") \
Рисунок 8 – Выполнение mlflow_integration.py \
7. В MLflow появился эксперимент Emotion-Classification-FineTuning с зафиксированными параметрами и метриками. \
![скриншот](report/Screenshot12.png "Рисунок") \
Рисунок 9 – Эксперимент в MLflow \

![скриншот](report/Screenshot13.png "Рисунок") \
Рисунок 10 – Все запуски скрипта \

![скриншот](report/Screenshot14.png "Рисунок") \
Рисунок 11 – Полученные метрики \

![скриншот](report/Screenshot15.png "Рисунок") \
Рисунок 12 – Параметры скрипта \

![скриншот](report/Screenshot16.png "Рисунок") \
Рисунок 13 – Модели метрик \

![скриншот](report/Screenshot17.png "Рисунок") \
Рисунок 14 –Содержимое вкладки Artifacts \
8. Проведён гиперпараметрический перебор (hyperparameter_tuning.py); запуски отображены в MLflow. \
![скриншот](report/Screenshot19.png "Рисунок") \
Рисунок 15 – Запуск hyperparameter_tuning.py \

![скриншот](report/Screenshot20.png "Рисунок") \
Рисунок 16 – Все запуски скрипта \

9. Выполнен анализ результатов (analyze_results.py); получены таблицы и визуализации по метрикам. \
![скриншот](report/Screenshot23-1.png "Рисунок") \
![скриншот](report/Screenshot23-2.png "Рисунок") \
Рисунок 17 – Анализ экспериментов \

##### Тестирование
- [x] Модульные тесты — не применялись.
- [x] Интеграционные тесты — проверены интеграции: HF Hub <-> Trainer <-> MLflow.
- [x] Производительность — учебная выборка; масштабирование требует отдельной настройки.
##### Выводы
Освоены основные механизмы работы с Hugging Face Hub: поиск, загрузка моделей и датасетов. Реализована тонкая настройка предобученной модели для текстовой классификации с использованием Transformers; устранены несовместимости параметров обучения. Интеграция с MLflow выполнена: гиперпараметры, метрики и артефакты логируются и доступны для анализа; выполнено сравнение запусков и HPO-исследование.
##### Приложения
- Скрипты: hf_hub_exploration.py, fine_tuning.py, mlflow_integration.py, hyperparameter_tuning.py, analyze_results.py.
- resources.txt с перечислением моделей/датасетов.
- Скриншоты результатов в папке report/
