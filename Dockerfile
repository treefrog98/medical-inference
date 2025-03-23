# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app code to the container
COPY . /app

# Download the model
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('microsoft/BioGPT-Large'); AutoModelForCausalLM.from_pretrained('microsoft/BioGPT-Large')"

# Expose the port the app will run on
EXPOSE 8000

# Command to run the FastAPI app with uvicorn
RUN python inference.py
