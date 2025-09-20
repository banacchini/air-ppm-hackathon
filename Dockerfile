# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by osmium
RUN apt-get update && apt-get install -y libexpat1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary files
COPY main.py .
COPY prediction.py .
COPY weather_preprocessing.py .
COPY models/ ./models/

# Default command: run the forecast script
ENTRYPOINT ["python", "main.py"]