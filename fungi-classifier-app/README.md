# Fungi CLassifier App
## Project Structure
```
fungi-classifier-app/
├── app/
│   ├── __init__.py          # Flask app initialization
│   ├── config.py            # Configuration settings
│   ├── models/
│   │   └── classifier.py    # Model inference logic
│   ├── api/
│   │   └── routes.py        # API endpoints
│   ├── utils/
│   │   ├── preprocessing.py # Image preprocessing
│   │   └── validation.py    # Input validation
│   ├── static/
│   │   ├── css/style.css   # UI styling
│   │   └── js/main.js      # Frontend logic
│   └── templates/
│       └── index.html      # Web interface
├── docker/
│   └── Dockerfile          # Container configuration
├── scripts/
│   └── download_model.sh   # Model download utility
└── requirements.txt        # Python dependencies
```
## Running the application

The application supports two modes of operation:
1. API Mode: Uses HuggingFace's inference API (requires HF_API_TOKEN)
2. Local Mode: Runs model inference locally (requires downloading model)

To run the application:

1. API Mode:
```bash
export FLASK_APP=app
export MODEL_TYPE=api
export HF_API_TOKEN=your_token
flask run
```
### Local Mode:
1. First download the model
```bash
./scripts/download_model.sh 70ziko/fungi-classifier
```
2. Initialize envvars and run the application
```bash
export FLASK_APP=app
export MODEL_TYPE=local
flask run
```
For containerized deployment:
```bash
docker build -t fungi-classifier -f docker/Dockerfile .
docker run -p 5000:5000 -e HF_API_TOKEN=your_token fungi-classifier
```