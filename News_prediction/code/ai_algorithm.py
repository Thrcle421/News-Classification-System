import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertConfig
from peft import PeftModel, PeftConfig


class NewsClassifier:
    def __init__(self):
        self.model_path = os.path.join(
            os.path.dirname(__file__), 'lora_final_model')
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        # AGNews categories
        self.categories = ['World', 'Sports', 'Business', 'Science/Tech']

    def load_model(self):
        """Load model and tokenizer"""
        try:
            # 加载基础模型配置
            config = DistilBertConfig.from_pretrained(
                'distilbert-base-uncased',
                num_labels=4
            )

            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                'distilbert-base-uncased')

            # 加载基础模型
            base_model = AutoModelForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                config=config
            )

            # 加载 LoRA 配置和权重
            peft_config = PeftConfig.from_pretrained(self.model_path)
            self.model = PeftModel.from_pretrained(base_model, self.model_path)

            self.model.to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            return False

    def predict(self, text):
        """Predict the category of a news article"""
        if self.model is None or self.tokenizer is None:
            if not self.load_model():
                return None, None

        try:
            # 准备输入
            inputs = self.tokenizer(text,
                                    return_tensors="pt",
                                    truncation=True,
                                    max_length=128,
                                    padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 进行预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(
                    outputs.logits, dim=-1)

            # 获取预测结果
            predicted_class = torch.argmax(predictions, dim=1).item()
            confidence = predictions[0][predicted_class].item() * 100

            return self.categories[predicted_class], confidence
        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            return None, None


# Create global classifier instance
classifier = NewsClassifier()
