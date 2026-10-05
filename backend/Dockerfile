FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app/backend

COPY requirements.txt ./
# PIP_INDEX_URL lets the deployment build pull wheels from a region-local mirror
# (Aliyun VPC-internal PyPI on ECS); the default keeps pypi.org for CI/overseas builds.
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir -i "${PIP_INDEX_URL}" -r requirements.txt

COPY app ./app
COPY config ./config
COPY run.py ./

EXPOSE 8000

# The public listener belongs to host Nginx; Compose publishes this port on
# 127.0.0.1 only and the application still uses its PostgreSQL checkpointer.
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8000"]
