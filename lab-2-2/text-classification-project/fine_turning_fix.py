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
