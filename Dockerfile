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

# pyproject.toml travels with the lock so the image carries the INTENTIONAL
# dependency declaration next to the pinned closure of it. Without it the
# image records what was installed but not what was asked for.
COPY requirements.txt pyproject.toml ./

# Install all Python deps into a user-local prefix so we can copy them cleanly
RUN pip install --no-cache-dir --user -r requirements.txt


# ─── Stage 2: runtime image ───────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Runtime system deps:
#   ffmpeg      — audio processing (replaces bundled ffmpeg.exe)
#   libsndfile1 — required by soundfile / Coqui TTS
#
# tesseract-ocr was here for "OCR fallback for scanned PDFs" — but the OCR
# engine is easyocr (document_engine.get_ocr_reader), and pytesseract is never
# imported anywhere in the repo. It was a dead apt layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
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

# NLTK corpora — baked for the same reason as the Whisper model above, but
# this one is not optional.
#
# `unstructured`, which the Word and Excel loaders use, downloads two corpora at
# IMPORT time unless told otherwise: unstructured/nlp/tokenize.py runs
# download_nltk_packages() at module scope whenever AUTO_DOWNLOAD_NLTK is unset
# or "true". So ingesting one .docx reaches the network from an application whose
# premise is that everything runs locally, and it does it through the nltk
# Downloader — the component CVE-2026-33236's path traversal was in.
#
# Note the absence of `|| true`, unlike the Whisper line. Whisper degrades: no
# model, no transcription, everything else works. These corpora do not — with
# the download switched off at runtime and the files missing, the first .docx
# raises LookupError from inside a loader. A build that cannot fetch them should
# fail here, where somebody is watching.
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN python -c "import nltk; \
    nltk.download('punkt_tab', download_dir='/usr/local/share/nltk_data'); \
    nltk.download('averaged_perceptron_tagger_eng', download_dir='/usr/local/share/nltk_data')"
ENV AUTO_DOWNLOAD_NLTK=false

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "main_discord.py"]
