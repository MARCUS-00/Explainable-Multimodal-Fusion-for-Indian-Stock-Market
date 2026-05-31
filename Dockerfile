FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        --prefer-binary -r requirements.txt && \
    pip install -e . && \
    find /usr/local/lib/python3.11 -type d -name __pycache__ -exec rm -rf {} + || true
COPY . .
RUN mkdir -p data/merged data/technical && \
    curl -L "https://github.com/MARCUS-00/Explainable-Multimodal-Fusion-for-Indian-Stock-Market/releases/download/v1.0-data/merged_final.csv" \
        -o data/merged/merged_final.csv && \
    curl -L "https://github.com/MARCUS-00/Explainable-Multimodal-Fusion-for-Indian-Stock-Market/releases/download/v1.0-data/technical.csv" \
        -o data/technical/technical.csv
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]