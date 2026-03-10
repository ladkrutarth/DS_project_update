# Use Python 3.9 slim as the base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p dataset/csv_data models scripts

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV VERISCAN_API_URL=http://backend:8000

# Expose ports
EXPOSE 8000
EXPOSE 8502

# Default command (can be overridden in docker-compose)
CMD ["python", "api/main.py"]
