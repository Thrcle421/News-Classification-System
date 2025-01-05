#!/usr/bin/env python
# coding: utf-8

# In[5]:


import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType


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

lora_metrics, lora_time = train_and_evaluate(lora_model, "lora", training_args)
lora_trainable_params = sum(p.numel()
                            for p in lora_model.parameters() if p.requires_grad)

print(f"LoRA - Trainable Parameters: {lora_trainable_params}")
print(f"LoRA - Training Time: {lora_time:.2f} seconds")
print(f"LoRA - Metrics: {lora_metrics}")


# In[ ]:




