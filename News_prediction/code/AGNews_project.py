#!/usr/bin/env python
# coding: utf-8

# In[5]:


import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
import os


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)


dataset = load_dataset("ag_news")
train_dataset = dataset["train"]
test_dataset = dataset["test"]


# In[ ]:


model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)
train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])
train_dataset.set_format("torch")
test_dataset.set_format("torch")


def train_and_evaluate(model, method_name, train_args):
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda p: {
            "accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()
        },
    )

    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    metrics = trainer.evaluate()
    model.save_pretrained(f"./{method_name}_final_model")
    return metrics, training_time


def initialize_model():
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        push_to_hub=False,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_lin", "k_lin", "v_lin"],
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=4)
    lora_model = get_peft_model(base_model, lora_config)

    return lora_model, training_args


def train_initial_model():
    lora_model, training_args = initialize_model()
    lora_metrics, lora_time = train_and_evaluate(
        lora_model, "lora", training_args)

    print(f"LoRA - Trainable Parameters: {sum(p.numel()
          for p in lora_model.parameters() if p.requires_grad)}")
    print(f"LoRA - Training Time: {lora_time:.2f} seconds")
    print(f"LoRA - Metrics: {lora_metrics}")

    lora_model.save_pretrained("./lora_final_model")


if __name__ == "__main__":
    train_initial_model()


# In[ ]:


def train_model_with_new_data(new_data):
    """Train the model with new data while preserving existing knowledge using LoRA"""
    model_path = os.path.join(os.path.dirname(__file__), 'lora_final_model')
    print('model_path', model_path)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    base_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=4
    )

    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_lin", "k_lin", "v_lin"],
    )

    model = get_peft_model(base_model, lora_config)

    if os.path.exists(model_path):
        print("Loading existing LoRA weights...")
        model = PeftModel.from_pretrained(base_model, model_path)

    category_to_id = {
        'World': 0,
        'Sports': 1,
        'Business': 2,
        'Science/Tech': 3
    }

    new_dataset = Dataset.from_dict({
        "text": [item["text"] for item in new_data],
        "label": [category_to_id[item["label"]] for item in new_data]
    })

    new_dataset = new_dataset.map(
        lambda examples: tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=128),
        batched=True
    )
    new_dataset = new_dataset.remove_columns(["text"])
    new_dataset.set_format("torch")

    training_args = TrainingArguments(
        output_dir=os.path.join(os.path.dirname(__file__), "results"),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_dir=os.path.join(os.path.dirname(__file__), "logs"),
        logging_steps=10,
        push_to_hub=False,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=new_dataset,
        tokenizer=tokenizer,
    )

    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning completed")

    print(f"Saving model and tokenizer to {model_path}")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    import json
    with open(os.path.join(model_path, 'category_mapping.json'), 'w') as f:
        json.dump(category_to_id, f)

    print("Model, tokenizer and category mapping saved successfully")


# In[ ]:
