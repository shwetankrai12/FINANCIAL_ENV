# Multi-stage build — lean final image
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Final image ──────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .

# HF Spaces runs as non-root
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# HF Spaces expects port 7860
ENV APP_PORT=7860
EXPOSE 7860

CMD ["python", "main.py"]