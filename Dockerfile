# Web version (sin Tkinter)
FROM python:3.10-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY . /app

# Install dependencies with platform-specific tensorflow for Linux
RUN uv pip install --system \
    "img2pdf>=0.6.3" \
    "matplotlib>=3.10.8" \
    "opencv-python-headless>=4.11.0.86" \
    "pandas>=2.3.3" \
    "pillow>=12.1.0" \
    "pydicom>=3.0.1" \
    "pytest>=9.0.2" \
    "tensorflow==2.15.0" \
    "numpy<2"

ENV PYTHONUNBUFFERED=1

EXPOSE 8000
ENV NEUMONIA_HOST=0.0.0.0
ENV NEUMONIA_PORT=8000

CMD ["python", "web.py"]
