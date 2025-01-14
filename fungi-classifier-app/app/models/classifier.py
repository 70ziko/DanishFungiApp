from abc import ABC, abstractmethod
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import requests
import os

class BaseClassifier(ABC):
    @abstractmethod
    def predict(self, image):
        pass

class HFAPIClassifier(BaseClassifier):
    def __init__(self, model_id, api_token):
        self.model_id = model_id
        self.api_token = api_token
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.headers = {"Authorization": f"Bearer {api_token}"}

    def predict(self, image):
        with open(image, "rb") as f:
            data = f.read()
        response = requests.post(self.api_url, headers=self.headers, data=data)
        return response.json()

class LocalClassifier(BaseClassifier):
    def __init__(self, model_path):
        self.model = AutoModelForImageClassification.from_pretrained(model_path)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        img = Image.open(image)
        inputs = self.feature_extractor(img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        return {
            "label": self.model.config.id2label[probs.argmax().item()],
            "score": probs.max().item()
        }

def get_classifier(config):
    if config.MODEL_TYPE == 'api':
        if not config.HF_API_TOKEN:
            raise ValueError("HF_API_TOKEN is required for API classifier")
        return HFAPIClassifier(config.HF_MODEL_ID, config.HF_API_TOKEN)
    else:
        if not os.path.exists(config.MODEL_PATH):
            raise ValueError(f"Model not found at {config.MODEL_PATH}")
        return LocalClassifier(config.MODEL_PATH)
