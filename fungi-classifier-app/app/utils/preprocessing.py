from PIL import Image
import os

def preprocess_image(image_path):
    """Preprocess image for model input"""
    try:
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if needed (adjust size based on model requirements)
        target_size = (224, 224)  # Standard size for many vision models
        if img.size != target_size:
            img = img.resize(target_size)
        
        # Save preprocessed image
        preprocessed_path = f"{os.path.splitext(image_path)[0]}_preprocessed.jpg"
        img.save(preprocessed_path, format='JPEG', quality=95)
        
        return preprocessed_path
        
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")
