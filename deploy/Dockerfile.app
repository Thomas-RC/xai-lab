FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/app/.cache/torch

WORKDIR /app

# WeasyPrint potrzebuje cairo + pango + gdk-pixbuf
# libgl1 dla OpenCV (zaleznosc pytorch-grad-cam)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    shared-mime-info \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# torch + torchvision z indeksu CUDA 12.4 — dziala z GPU (przy `--gpus all`)
# i z CPU (gdy GPU niedostepne). Wieksze ~3 GB ale daje wybor user-side.
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.5.*" \
    "torchvision==0.20.*"

# Reszta zaleznosci
COPY pyproject.toml ./
RUN pip install \
    "numpy>=1.26.0,<2.2" \
    "pillow>=11.0.0" \
    "captum>=0.7.0" \
    "grad-cam>=1.5.0" \
    "lime>=0.2.0.1" \
    "scikit-image>=0.24.0" \
    "streamlit>=1.40.0" \
    "matplotlib>=3.9.0" \
    "pydantic>=2.9.0" \
    "pydantic-settings>=2.6.0" \
    "structlog>=24.4.0" \
    "python-dotenv>=1.0.0" \
    "weasyprint>=63.0" \
    "markdown>=3.7" \
    "pygments>=2.18.0" \
    "pandas>=2.2.0" \
    "tabulate>=0.9.0"

COPY src ./src
COPY scripts ./scripts
COPY data ./data
COPY report ./report

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "src/ui/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
