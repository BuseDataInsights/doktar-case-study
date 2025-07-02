FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY blackfly_pipeline.py .
COPY blackfly.yaml .

# Create necessary directories
RUN mkdir -p images/train images/val images/test1 images/test2 \
    labels/train labels/val \
    runs/detect

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "blackfly_pipeline.py"] 