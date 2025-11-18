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

