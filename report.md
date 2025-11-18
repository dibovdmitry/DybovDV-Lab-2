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
- [x] Создать скрипт hf_hub_exploration.py - скачать модель/датасет и проверить содержимое.
- [x] Подготовить проект и файл resources.txt с информацией о выбранных ресурсах.
- [x] Создать fine_tuning.py (использован параметр evaluation_during_training при необходимости).
- [x] Запустить MLflow Tracking Server и интегрировать обучение через mlflow_integration.py.
- [x] Провести гиперпараметрическое исследование с hyperparameter_tuning.py.
- [x] Проанализировать результаты в analyze_results.py.
##### Ключевые фрагменты кода
- Скрипт hf_hub_exploration.py:
```python
from datasets import load_dataset
from huggingface_hub import list_models, list_datasets
import pandas as pd

print("Available datasets for text classification:")
datasets = list_datasets(filter="task_categories:text-classification")
for dataset in datasets:
	print(f"- {dataset.id}")

print("\nUloading dataset emotion...")
dataset = load_dataset("emotion")

print(f"\nDataset structure: {dataset}")
print(f"\nExamples from train split:")
train_df = pd.DataFrame(dataset['train'][:5])
print(train_df)

print("\nClass distribution in the training data:")
label_counts = pd.Series(dataset['train']['label']).value_counts()
print(label_counts)

print("\n\nAvailable models for text classification:")
models = list_models(
	filter="task:text-classification",
	sort="downloads",
	direction=-1,
	limit=5
)

for model in models:
	print(f"\nModel: {model.id}")
	print(f"\nDownloads: {model.downloads}")
	print(f"\nTags: {model.tags}")
	if model.pipeline_tag:
		print(f"Task type: {model.pipeline_tag}")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased"
print(f"\nLoading the model {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
	model_name,
	num_labels=6 #Num of classes in dataset
)

print("The model and tokenizer have been successfully loaded!")
print(f"Dictionary size: {tokenizer.vocab_size}")
print(f"Model architecture: {model.__class__.__name__}")

test_text = "i am feeling grouchy"
print(f"Text for test: {test_text}")

tokens = tokenizer(test_text, return_tensors="pt")
print(f"Tokens: {tokens}")
print(f"Decoration tokens: {tokenizer.decode(tokens['input_ids'][0])}")
```
- Скрипт fine_turning.py:
```python
from datasets import load_dataset
from transformers import (
	AutoTokenizer,
	AutoModelForSequenceClassification,
	TrainingArguments,
	Trainer,
	DataCollatorWithPadding
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch

dataset = load_dataset("emotion")

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def tokenize_function(examples):
	return tokenizer(
		examples["text"],
		truncation=True,
		padding=True,
		max_length=128
	)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

id2label = { 0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForSequenceClassification.from_pretrained(
	model_name,
	num_labels=6,
	id2label=id2label,
	label2id=label2id
)

def compute_metrics(eval_pred):
	predictions, labels = eval_pred
	predictions = np.argmax(predictions, axis=1)
	acc = accuracy_score(labels, predictions)
	f1 = f1_score(labels, predictions, average="weighted")	
	return {"accuracy": acc, "f1_score": f1}

training_args = TrainingArguments(
	output_dir="./results",
	learning_rate=2e-5,
	per_device_train_batch_size=16,
	per_device_eval_batch_size=16,
	num_train_epochs=3,
	weight_decay=0.01,
	eval_strategy='epoch',
	save_strategy="epoch",
	load_best_model_at_end=True,
	metric_for_best_model="f1_score",
	logging_dir="./logs",
	logging_steps=100,
	report_to="none"
)

trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=tokenized_datasets["train"],
	eval_dataset=tokenized_datasets["validation"],
	tokenizer=tokenizer,
	data_collator=data_collator,
	compute_metrics=compute_metrics,
)

print("The beginning of training...")
train_result = trainer.train()
trainer.save_model("./emotion-classifier")
trainer.save_metrics("train", train_result.metrics)
print("Training completed!")

print("Results based on test data:")
test_results = trainer.evaluate(tokenized_datasets["test"])
print(test_results)


with open("test_results.txt", "w") as f:
	f.write(f"Accuracy: {test_results['eval_accuracy']:.4f}\n")
	f.white(f"F1 Score: {test_results['eval_f1_score']:.3f}\n")

def predict_emotion(text):
	inputs = tokenizer(text, return_tensors="pt", truncation=True)
	model.eval()
	with torch.no_grad():
		outputs = model(**inputs)
		probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
		pred_idx = int(torch.argmax(probs, dim=1).item())
		return model.config.id2label[pred_idx], float(probs[0, pred_idx].item())

test_text = [
	"I am feel sadness in the bottom of my soul...",
	"This is making me fear my future life",
	"I'm so frustrated and angry about my boring life"
]

print("\nTestiong of model:")
for text in test_text:
	emotion, confidence = predict_emotion(text)
	print(f"Text: '{text}'")
	print(f"Prediction: {emotion} (confidence: {confidence:.3f})")
	print()
```
- Скрипт mlflow_integration.py:
```python
import mlflow 
import mlflow.transformers 
from datasets import load_dataset 
from transformers import ( 
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding, 
) 
from transformers.trainer_utils import EvaluationStrategy 
import numpy as np 
from sklearn.metrics import accuracy_score, f1_score 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 6

mlflow.set_tracking_uri("http://localhost:5000") 
mlflow.set_experiment("Emotion-Classification-FineTuning") 

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred): 
    """Вычисляет метрики Accuracy и F1-score."""
    predictions, labels = eval_pred 
    predictions = np.argmax(predictions, axis=1) 
    acc = accuracy_score(labels, predictions) 
    # average="weighted" используется для несбалансированных классов
    f1 = f1_score(labels, predictions, average="weighted") 
    return {"eval_accuracy": acc, "eval_f1_score": f1} 


def tokenize_function(examples): 
    """Токенизирует текст."""
    return tokenizer( 
        examples["text"], 
        truncation=True, 
        padding=True, 
        max_length=128 
    )
    

print("Загрузка и токенизация данных (происходит один раз)...")
try:
    dataset = load_dataset("emotion") 
    tokenized_datasets = dataset.map(tokenize_function, batched=True) 
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels") 
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"]) 
except Exception as e:
    print(f"Ошибка при загрузке или подготовке данных: {e}")
    tokenized_datasets = None


def train_model(learning_rate, batch_size=16, num_epochs=3, weight_decay=0.01):
    """
    Инициализирует, обучает и оценивает модель с заданным learning rate.
    
    Args:
        learning_rate (float): Скорость обучения для Trainer.
        
    Returns:
        dict: Словарь с итоговыми метриками оценки (ТОЛЬКО ЧИСЛОВЫЕ ЗНАЧЕНИЯ).
    """
    if tokenized_datasets is None:
        return {"error": "Data not loaded"}

    model_params = { 
        "model_name": MODEL_NAME, 
        "num_labels": NUM_LABELS, 
        "learning_rate": learning_rate, # Используем переданный параметр
        "batch_size": batch_size, 
        "num_epochs": num_epochs, 
        "weight_decay": weight_decay 
    } 
    
    model = AutoModelForSequenceClassification.from_pretrained( 
        model_params["model_name"], 
        num_labels=model_params["num_labels"], 
        # Добавляем id2label/label2id для корректного сохранения
        id2label={0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}, 
        label2id={'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5} 
    ) 


    training_args = TrainingArguments( 
        output_dir=f"./results/lr_{learning_rate}", # Уникальная папка для каждого LR
        learning_rate=learning_rate, 
        per_device_train_batch_size=batch_size, 
        per_device_eval_batch_size=batch_size, 
        num_train_epochs=num_epochs, 
        weight_decay=weight_decay, 
        eval_strategy=EvaluationStrategy.EPOCH, # Оценка в конце каждой эпохи
        save_strategy="epoch",                   # Сохранение в конце каждой эпохи
        load_best_model_at_end=True,             # Загрузка лучшей модели в конце
        metric_for_best_model="eval_f1_score",   # Выбор лучшей модели по F1-score
        
        logging_dir="./logs", 
        logging_steps=100, 
        report_to="none", 
        seed=42
    ) 
    
    trainer = Trainer( 
        model=model, 
        args=training_args, 
        train_dataset=tokenized_datasets["train"], 
        eval_dataset=tokenized_datasets["validation"], 
        tokenizer=tokenizer, 
        data_collator=data_collator, 
        compute_metrics=compute_metrics, 
    ) 

    print(f"Начало обучения для LR: {learning_rate}")
    trainer.train() 

    test_results = trainer.evaluate(tokenized_datasets["test"], metric_key_prefix="test")
    
    model_path = f"./best_model_lr_{learning_rate}"
    trainer.save_model(model_path)
    
    mlflow.transformers.log_model( 
        transformers_model={ 
            "model": model, 
            "tokenizer": tokenizer 
        }, 
        artifact_path="emotion-classifier", 
        registered_model_name="distilbert-emotion-classifier",
        task="text-classification" # <-- Добавлено для явного указания задачи
    ) 
    
    if trainer.state.best_model_checkpoint:
        mlflow.set_tag("best_checkpoint_path", trainer.state.best_model_checkpoint)
        
    final_metrics = {
        **test_results,
        "best_validation_f1": trainer.state.best_metric,
    }
    
    return final_metrics
```
- Скрипт hyperparameter.py:
```python

```
- Скрипт analyze_resylts.py:
```python
import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "Emotion-Classification-FineTuning"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

print(f"Подключение к MLflow на URI: {TRACKING_URI}")
print(f"Поиск эксперимента: '{EXPERIMENT_NAME}'")

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    print(f"\n❌ ОШИБКА: Эксперимент '{EXPERIMENT_NAME}' не найден.")
    print("Возможные причины:")
    print("1. MLflow Tracking Server (mlflow ui) не запущен по адресу http://localhost:5000.")
    print("2. В файле 'hyperparameter_tuning.py' еще не было запусков, которые бы создали этот эксперимент.")
    
    try:
        experiment_id = client.create_experiment(EXPERIMENT_NAME)
        experiment = client.get_experiment(experiment_id)
        print(f"Эксперимент '{EXPERIMENT_NAME}' был создан, но пока пуст.")
    except Exception as e:
        print(f"Не удалось создать эксперимент: {e}")
        # Завершаем скрипт, если не удалось ни найти, ни создать
        exit(1)

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
```
##### Результаты выполнения
1. Установлены пакеты huggingface_hub, datasets, transformers, pandas, numpy. \
![скриншот](report/Screenshot1.png "Рисунок") \
Рисунок 1 – Установка пакетов 
2. На Hugging Face Hub найден подходящий репозиторий модели и изучена модельная карточка. \
![скриншот](report/Screenshot2.png "Рисунок") \
Рисунок 2 – Найденная модель на Hugging Face Hub 
3. Создан hf_hub_exploration.py, выполнено скачивание модели. \
![скриншот](report/Screenshot4.png "Рисунок") \
Рисунок 3 – Выполнение hf_hub_exploration.py 
4. Создана папка проекта и resources.txt с записями выбранных ресурсов. \
![скриншот](report/Screenshot5.png "Рисунок") \
Рисунок 4 – Создание проекта и resources.txt \
![скриншот](report/Screenshot6.png "Рисунок") \
Рисунок 5 – Проверка содержимого resources.txt 
5. Создан fine_tuning.py, в коде использован параметр eval_training вместо evaluation_training, запущен тренинг. \
![скриншот](report/Screenshot8-1.png "Рисунок") \
![скриншот](report/Screenshot8-2.png "Рисунок") \
Рисунок 6 – Запуск fine_tuning.py 
7. Запущен MLflow Tracking Server, произведена интеграция через mlflow_integration.py. \
![скриншот](report/Screenshot9.png "Рисунок") \
Рисунок 7 – MLflow Tracking Server 
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
Рисунок 14 –Содержимое вкладки Artifacts
8. Проведён гиперпараметрический перебор (hyperparameter_tuning.py); запуски отображены в MLflow. \
![скриншот](report/Screenshot19.png "Рисунок") \
Рисунок 15 – Запуск hyperparameter_tuning.py \
![скриншот](report/Screenshot20.png "Рисунок") \
Рисунок 16 – Все запуски скрипта
9. Выполнен анализ результатов (analyze_results.py), получены таблицы и визуализации по метрикам. \
![скриншот](report/Screenshot23-1.png "Рисунок") \
![скриншот](report/Screenshot23-2.png "Рисунок") \
Рисунок 17 – Анализ экспериментов

##### Тестирование
- [x] Модульные тесты - не применялись.
- [x] Интеграционные тесты - проверены интеграции: HF Hub <-> Trainer <-> MLflow.
- [x] Производительность - учебная выборка; масштабирование требует отдельной настройки.
##### Выводы
Освоены основные механизмы работы с Hugging Face Hub: поиск, загрузка моделей и датасетов. Реализована тонкая настройка предобученной модели для текстовой классификации с использованием Transformers; устранены несовместимости параметров обучения. Интеграция с MLflow выполнена: гиперпараметры, метрики и артефакты логируются и доступны для анализа; выполнено сравнение запусков.
##### Приложения
- Скрипты: hf_hub_exploration.py, fine_tuning.py, mlflow_integration.py, hyperparameter_tuning.py, analyze_results.py.
- resources.txt с перечислением моделей/датасетов.
- Скриншоты результатов в папке report/
