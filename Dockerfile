# Stage 1: Build dependencies (Cài đặt dependencies trong image build tạm thời)
FROM python:3.9-slim AS build

# Set environment variables to prevent interactive prompts
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements.txt to the container
COPY requirements.txt /app/

# Install system dependencies needed for TensorFlow and others
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Stage 2: Final image (Chạy ứng dụng và cắt bỏ các file build không cần thiết)
FROM python:3.9-slim

# Set environment variables to prevent interactive prompts
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only the necessary files from the build stage (dependencies and libraries)
COPY --from=build /usr/local /usr/local

# Copy the application code into the container
COPY . /app/

# Expose port 8000 for the app
EXPOSE 8000

WORKDIR /app/src

# Command to run the FastAPI app
CMD ["uvicorn", "application:app", "--host", "0.0.0.0", "--port", "8000"]
