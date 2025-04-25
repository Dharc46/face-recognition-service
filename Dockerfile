# Stage 1: Build dependencies
FROM python:3.9-slim AS build

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Cài gói cần thiết và xoá ngay sau khi dùng để giảm size
COPY requirements.txt ./ 

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove build-essential gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache

# Stage 2: Final minimal image
FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Sao chép môi trường Python đã cài từ image build
COPY --from=build /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Sao chép mã nguồn ứng dụng
COPY . /app/

# Loại trừ thư mục không cần thiết trong container
RUN rm -rf /app/tests /app/docs /app/notebooks && \
    rm -rf /app/.git /app/__pycache__ /app/*.pyc /app/*.pyo /app/*.pyd

# Expose cổng dịch vụ
EXPOSE 8000

WORKDIR /app/src

# Khởi chạy ứng dụng FastAPI
CMD ["uvicorn", "application:app", "--host", "0.0.0.0", "--port", "8000"]
