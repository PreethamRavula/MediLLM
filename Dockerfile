# 1) Small base image with Python 3.10
FROM python:3.10-slim

# 2) System libs that Pillow/OpenCV/ffmpeg need at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsm6 libxext6 libgl1 curl \
 && rm -rf /var/lib/apt/lists/*

# 3) Faster, cleaner Python logs + consistent HF cache location
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 4) Working directory
WORKDIR /app

# 5) Install Python deps first (better build caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 6) Copy your soucre code (done after deps so earlier layers cache)
COPY . .

# 7) Env for Gradio + Matplotlib
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    MPLCONFIGDIR=/tmp/mpl

# 7) Expose the UI port
EXPOSE 7860

# 8) Start your Gradio demo
CMD ["python", "app/demo/demo.py"]
