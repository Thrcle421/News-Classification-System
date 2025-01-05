# Fine-Tuning for Text Classification using Low-Rank Adaptation (LoRA)

### Comparations

The code of comparing LoRA fine-tuning with full-parameter fine-tuning is in two ipynb files. Because when I was training, sometimes there were mistakes in the code, so I wanted to implement the project in each part.

### LoRA+

The LoRA+ implementation is in the lora_plus file, you can get the original code using the following

```
git clone https://github.com/nikhil-ghosh-berkeley/loraplus.git
cd loraplus
pip install -r requirements.txt
```

Then you can use

```python
python LoRA_plus_classification.py
```

to implement the LoRA+ experiment in the paper.
