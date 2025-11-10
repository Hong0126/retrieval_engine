# -------- base --------
FROM python:3.11-slim

# 基础环境
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AIP_HEALTH_ROUTE=/health \
    AIP_PREDICT_ROUTE=/predict \
    # PyTorch 显存碎片优化（可按需调小/调大）
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# 运行时依赖（opencv 等）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

# 升级打包工具
RUN python -m pip install --upgrade pip setuptools wheel

# ---- 深度学习（GPU）----
# 直接安装 GPU 版 torch/torchvision（CUDA 12.1 对应 cu121 轮）
# 注意：不要先装 CPU 版 torch，以免被锁定在 CPU
RUN pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu

# 其余依赖；同样带上 cu121 索引，防止 requirements 里的 torch/torchvision 被回装为 CPU 轮
COPY requirements.txt /tmp/requirements.txt 
RUN pip install --no-cache-dir -r /tmp/requirements.txt
# 拷贝服务代码（确保 app/index/* 已就绪）
WORKDIR /app
COPY ./app /app

# （可选）构建期预热，失败忽略
RUN python -u /app/warmup.py || true

EXPOSE 8080
# 1 worker 即可，避免重复加载模型
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
