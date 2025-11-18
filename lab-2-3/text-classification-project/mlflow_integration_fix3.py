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
