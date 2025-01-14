from abc import ABC, abstractmethod
import torch
import timm
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
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
        # Load model using timm
        self.model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2000)
        
        # Load state dict
        state_dict = torch.load(os.path.join(model_path, 'model.pth'), map_location='cpu')
        self.model.load_state_dict(state_dict)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Setup preprocessing pipeline
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    def predict(self, image_path):
        # Read image using OpenCV
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        transformed = self.transform(image=image)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
        # Get prediction
        pred_idx = probs[0].argmax().item()
        confidence = probs[0].max().item()
        
        # Map index to label (you'll need to provide a mapping file)
        # For now, return the index
        return {
            "label": f"class_{pred_idx}",
            "score": confidence
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
