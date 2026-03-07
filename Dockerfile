# Stage 1: Build stage (Cài đặt dependencies và build libraries)
FROM python:3.11-slim as builder

# Cấu hình tối ưu môi trường cho Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho việc compile (ví dụ C extensions nếu có)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Tạo và kích hoạt môi trường ảo (virtual environment)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Sử dụng cache mount của Docker BuildKit giúp tiết kiệm thời gian tải file wheel/tar khi build lại
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Tải spaCy model trực tiếp vào môi trường ảo
RUN python -m spacy download en_core_web_sm

# ========================================== #

# Stage 2: Runtime stage (Chạy ứng dụng)
FROM python:3.11-slim

# Cấu hình môi trường tương tự builder và trỏ PATH tới môi trường ảo
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy toàn bộ môi trường ảo đã build từ Stage 1 sang Stage 2
# Nhờ vậy, image cuối cùng rất nhẹ, không bị dính apt cache, curl hay build-essential
COPY --from=builder /opt/venv /opt/venv

# THỰC HÀNH TỐT VỀ BẢO MẬT: Không được dùng quyền root trừ khi rất cần thiết
RUN addgroup --system appgroup && adduser --system --group appuser

# Copy toàn bộ source code của app vào và bàn giao quyền sở hữu cho appuser
COPY --chown=appuser:appgroup . .

# Chuyển sang user non-root vừa tạo
USER appuser

EXPOSE 8010

# Chạy server ứng dụng
# Lưu ý: Khi chạy ở production, có thể bạn sẽ muốn bật thêm vài worker để tối ưu hiệu suất,
# ví dụ CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010", "--workers", "4"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
