# Sử dụng image Python chính thức làm base image
FROM python:3.10-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Cài đặt các dependencies hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Sao chép file requirements.txt và cài đặt các thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Mở port 8000 để truy cập ứng dụng FastAPI
EXPOSE 8000

WORKDIR /app/src

# Lệnh chạy ứng dụng khi container khởi động
CMD ["uvicorn", "src.application:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]