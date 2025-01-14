import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

def preprocess_image(image_path):
    """Preprocess image for model input using the same transformations as training"""
    try:
        # Read image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise Exception("Failed to read image")
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Define preprocessing pipeline
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Apply transformations
        transformed = transform(image=img)
        processed_img = transformed['image']
        
        # Save preprocessed image
        preprocessed_path = f"{os.path.splitext(image_path)[0]}_preprocessed.jpg"
        cv2.imwrite(preprocessed_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
        
        return preprocessed_path
        
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")
