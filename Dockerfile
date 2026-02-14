# Web version (sin Tkinter)
FROM python:3.10-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY . /app

ENV UV_NO_DEV=1
RUN uv sync
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
ENV NEUMONIA_HOST=0.0.0.0
ENV NEUMONIA_PORT=8000

CMD ["python", "web.py"]
