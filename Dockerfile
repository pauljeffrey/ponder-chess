# Ponder backend (CPU inference). Build: docker build -t ponder .   Run: docker run -p 8000:8000 ponder
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements-serve.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements-serve.txt
COPY chessagent ./chessagent
COPY server ./server
COPY scripts/fetch_weights.py ./scripts/fetch_weights.py
COPY web ./web
COPY models.json ./models.json
COPY weights ./weights
COPY docker-entrypoint.sh ./
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/api/health')"
ENTRYPOINT ["sh", "docker-entrypoint.sh"]
