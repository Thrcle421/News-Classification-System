#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)


dataset = load_dataset("imdb")
train_dataset = dataset["train"].shuffle(seed=42)
test_dataset = dataset["test"].shuffle(seed=42)

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
    model_name, num_labels=2)
lora_model = get_peft_model(base_model, lora_config)

lora_metrics, lora_time = train_and_evaluate(lora_model, "lora", training_args)
lora_trainable_params = sum(p.numel()
                            for p in lora_model.parameters() if p.requires_grad)

full_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2)

full_metrics, full_time = train_and_evaluate(full_model, "full", training_args)
full_trainable_params = sum(p.numel()
                            for p in full_model.parameters() if p.requires_grad)

print(f"LoRA - Trainable Parameters: {lora_trainable_params}")
print(f"Full Fine-Tuning - Trainable Parameters: {full_trainable_params}")
print(f"LoRA - Training Time: {lora_time:.2f} seconds")
print(f"Full Fine-Tuning - Training Time: {full_time:.2f} seconds")
print(f"LoRA - Metrics: {lora_metrics}")
print(f"Full Fine-Tuning - Metrics: {full_metrics}")


# In[ ]:


import matplotlib.pyplot as plt
import os


def visualize(lora_time, full_time, lora_trainable_params, full_trainable_params, lora_metrics, full_metrics):
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.bar(["LoRA", "Full Fine-Tuning"],
            [lora_trainable_params, full_trainable_params], log=True)
    plt.ylabel("Trainable Parameters (log scale)")
    plt.title("Trainable Parameters Comparison")
    plt.savefig(os.path.join(
        output_dir, "trainable_parameters_comparison.png"))
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.bar(["LoRA", "Full Fine-Tuning"], [lora_time, full_time])
    plt.ylabel("Training Time (seconds)")
    plt.title("Training Time Comparison")
    plt.savefig(os.path.join(output_dir, "training_time_comparison.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.bar(["LoRA", "Full Fine-Tuning"],
            [lora_metrics["eval_loss"], full_metrics["eval_loss"]])
    plt.ylabel("Evaluation Loss")
    plt.title("Evaluation Loss Comparison")
    plt.savefig(os.path.join(output_dir, "evaluation_loss_comparison.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.bar(["LoRA", "Full Fine-Tuning"],
            [lora_metrics["eval_accuracy"], full_metrics["eval_accuracy"]])
    plt.ylabel("Accuracy")
    plt.ylim(0.5, 1.0)  
    plt.title("Accuracy Comparison")
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))
    plt.close()

    print(f"Visualizations saved in directory: {output_dir}")


# In[21]:


visualize(lora_time, full_time, lora_trainable_params,
          full_trainable_params, lora_metrics, full_metrics)


# In[ ]:


results = []
for r in [4, 8, 16]:
    for alpha in [16, 32, 64]:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r,
            lora_alpha=alpha,
            lora_dropout=0.1,
            target_modules=["q_lin", "k_lin", "v_lin"],
        )
        lora_model = get_peft_model(base_model, lora_config)
        metrics, time_taken = train_and_evaluate(
            lora_model, f"lora_r{r}_alpha{alpha}", training_args)
        results.append((r, alpha, metrics, time_taken))

for result in results:
    r, alpha, metrics, time_taken = result
    print(
        f"LoRA r={r}, alpha={alpha}: Accuracy={metrics['eval_accuracy']:.4f}, Time={time_taken:.2f}s")


# In[13]:


get_ipython().system('pip install seaborn')


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
results_df = pd.DataFrame(
    results, columns=["r", "alpha", "metrics", "time_taken"])
results_df["accuracy"] = results_df["metrics"].apply(
    lambda x: x["eval_accuracy"])
results_df["loss"] = results_df["metrics"].apply(lambda x: x["eval_loss"])

plt.figure(figsize=(10, 8))
accuracy_pivot = results_df.pivot(
    index="r", columns="alpha", values="accuracy")
sns.heatmap(accuracy_pivot, annot=True, fmt=".4f",
            cmap="YlGnBu", cbar_kws={"label": "Accuracy"})
plt.title("Accuracy")
plt.xlabel("LoRA Alpha")
plt.ylabel("LoRA r")
plt.savefig("accuracy.png")
plt.show()

plt.figure(figsize=(10, 8))
time_pivot = results_df.pivot(index="r", columns="alpha", values="time_taken")
sns.heatmap(time_pivot, annot=True, fmt=".2f", cmap="YlOrRd",
            cbar_kws={"label": "Training Time (s)"})
plt.title("Training Time")
plt.xlabel("LoRA Alpha")
plt.ylabel("LoRA r")
plt.savefig("time.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="time_taken",
    y="accuracy",
    hue="r",
    style="alpha",
    data=results_df,
    palette="viridis",
    s=100
)
plt.title("Accuracy && Training Time")
plt.xlabel("Training Time (s)")
plt.ylabel("Accuracy")
plt.legend(title="LoRA r")
plt.savefig("accuracy_and_time.png")
plt.show()

