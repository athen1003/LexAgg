FROM python:3.12-slim

WORKDIR /app

# 基础系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# 1. 装非 torch 依赖（requirements.txt 会顺带拉 CPU 版 torch）
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 -r requirements.txt

# 2. 按构建参数覆盖 torch（GPU 时换 CUDA 版）
ARG TORCH_INDEX=""
RUN if [ -n "$TORCH_INDEX" ]; then \
        pip install --no-cache-dir --force-reinstall torch==2.11.0 --index-url "$TORCH_INDEX"; \
    fi

# 国内镜像加速（可选）:
#   docker build --build-arg TORCH_INDEX=https://mirrors.aliyun.com/pytorch-wheels/cu128 \
#     --build-arg PIP_INDEX=https://mirrors.aliyun.com/pypi/simple .

# 3. 拷贝代码（放最后以利用 layer cache）
COPY app/ ./app/
COPY static/ ./static/
COPY data/ ./data/

RUN mkdir -p /app/models

EXPOSE 8000

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
