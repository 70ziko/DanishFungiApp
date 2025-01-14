FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p uploads
RUN mkdir -p app/models/fungi-classifier

# Set environment variables
ENV FLASK_APP=app
ENV FLASK_ENV=production
ENV MODEL_TYPE=local
ENV PYTHONPATH=/app

# Download model if needed (can be commented out if using API mode)
# RUN ./scripts/download_model.sh $HF_MODEL_ID

# Expose port
EXPOSE 5000

# Run the application
CMD ["flask", "run", "--host=0.0.0.0"]
