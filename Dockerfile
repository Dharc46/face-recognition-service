FROM python:3.10-slim

# Use Python 3.10 instead of 3.12 because TensorFlow doesn't fully support 3.12 yet

WORKDIR /app

# Install system dependencies including OpenCV requirements
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Explicitly install uvicorn to make sure it's available
RUN pip install --no-cache-dir uvicorn

# Copy application code
COPY . .

# Make sure the directories exist
RUN mkdir -p Dataset/FaceData/raw Dataset/FaceData/processed Models src/align

# Download Haar cascade for face detection backup
RUN wget -P src/ https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using python module format to avoid PATH issues
CMD ["python", "-m", "uvicorn", "application:app", "--host", "0.0.0.0", "--port", "8000"]