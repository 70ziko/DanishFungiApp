from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
from app.utils.validation import allowed_file
from app.utils.preprocessing import preprocess_image
from app.models.classifier import get_classifier

api = Blueprint('api', __name__)
classifier = None

@api.before_app_first_request
def initialize_classifier():
    global classifier
    classifier = get_classifier(current_app.config)

@api.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if not allowed_file(file.filename, current_app.config['ALLOWED_EXTENSIONS']):
        return jsonify({'error': 'File type not allowed'}), 400
        
    filename = secure_filename(file.filename)
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Preprocess image
        preprocessed_path = preprocess_image(filepath)
        
        # Get prediction
        result = classifier.predict(preprocessed_path)
        
        # Clean up
        os.remove(filepath)
        os.remove(preprocessed_path)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
