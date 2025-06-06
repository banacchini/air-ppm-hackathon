# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the forecasting script into the container
COPY main.py .


# Install system dependencies required by osmium
RUN apt-get update && apt-get install -y libexpat1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Default command: run the forecast script.
# It will pick up the --data-file, --landuse-pbf, and --output-file args from Docker run.
ENTRYPOINT ["python", "main.py"]
