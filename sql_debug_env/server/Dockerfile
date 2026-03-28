FROM python:3.11-slim

ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/sql_debug_env
RUN pip install --no-cache-dir /app/sql_debug_env

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn sql_debug_env.server.app:app --host 0.0.0.0 --port $PORT"]
