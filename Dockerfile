# Use a lightweight Python base image (Linux based)
FROM python:3.13.9-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc
# PYTHONUNBUFFERED: Prevents Python from buffering stdout and stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for Ollama (specifically curl)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama CLI
# This script detects the architecture and installs the correct binary
RUN curl -fsSL https://ollama.com/install.sh | sh

# --- Virtual Environment Setup ---
# Instead of copying the local .venv (which often fails due to OS mismatch),
# we create a fresh venv inside the container at /opt/venv.
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV

# Update PATH so that commands like `python` and `pip` automatically
# use the virtual environment, just like activation.
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies into the virtual environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Default command
# Ensure you serve Ollama if your app needs the server running locally
# CMD ["sh", "-c", "ollama serve & python main.py"]

# Or just run your python script:
CMD ["python", "main_discord.py"]