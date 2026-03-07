# ============================================ #
# Stage 1: Build – cài đặt dependencies
# ============================================ #
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Cài build tools cần thiết (cho các C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cài uv – trình quản lý package Python cực nhanh, giải quyết dependency tốt hơn pip
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Tạo virtual environment
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Copy requirements trước (tận dụng Docker layer cache)
COPY requirements.txt .

# Cài đặt dependencies bằng uv (nhanh hơn pip 10-100x, giải quyết dependency tốt hơn)
RUN uv pip install -r requirements.txt

# Tải spaCy model (dùng uv thay vì spacy download vì spacy download gọi pip nội bộ)
RUN uv pip install en_core_web_sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# ============================================ #
# Stage 2: Runtime – chạy ứng dụng
# ============================================ #
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy virtual environment đã build từ Stage 1
COPY --from=builder /opt/venv /opt/venv

# Tạo user non-root để tăng bảo mật
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Copy source code
COPY --chown=appuser:appgroup . .

USER appuser

EXPOSE 8010

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
