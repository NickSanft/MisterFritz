# ─── Stage 1: dependency builder ──────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Build-time system deps (compilers for packages with C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install all Python deps into a user-local prefix so we can copy them cleanly
RUN pip install --no-cache-dir --user -r requirements.txt


# ─── Stage 2: runtime image ───────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Runtime system deps:
#   ffmpeg      — audio processing (replaces bundled ffmpeg.exe)
#   libsndfile1 — required by soundfile / Coqui TTS
#   tesseract   — OCR fallback for scanned PDFs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application source (see .dockerignore for exclusions)
COPY . .

# Pre-create runtime directories expected by the application
RUN mkdir -p input output chroma_store temp_images temp_audio

# Use system ffmpeg (installed above) — overrides Windows .exe paths
ENV FFMPEG_PATH=ffmpeg
ENV FFPROBE_PATH=ffprobe

# Whisper STT — pre-bake the model into the image so first-run doesn't download it
ENV WHISPER_MODEL=small
ENV WHISPER_DEVICE=cpu
ENV WHISPER_COMPUTE_TYPE=int8
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')" || true

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "main_discord.py"]
