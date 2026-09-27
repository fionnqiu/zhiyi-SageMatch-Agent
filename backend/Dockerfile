FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app/backend

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY config ./config
COPY run.py ./

EXPOSE 8000

# The public listener belongs to host Nginx; Compose publishes this port on
# 127.0.0.1 only and the application still uses its PostgreSQL checkpointer.
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8000"]
