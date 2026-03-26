FROM python:3.11-slim AS base

WORKDIR /app

# Install system deps: gcc for building, texlive for PDF reports
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        texlive-latex-base \
        texlive-latex-extra \
        texlive-latex-recommended \
        texlive-fonts-recommended \
        texlive-science \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (cached layer)
COPY pyproject.toml ./
RUN pip install --no-cache-dir ".[frontend]"

# Copy source
COPY src/ src/
COPY frontend/ frontend/
COPY tests/ tests/

# ── API target ──────────────────────────────────────────────
FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Frontend target ─────────────────────────────────────────
FROM base AS frontend
EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
