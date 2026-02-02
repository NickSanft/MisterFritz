import torch
from TTS.api import TTS
import torchaudio
import subprocess
import re
import os

print("Hi")

# Get FFmpeg version without torchcodec
try:
    # On Windows, try common locations and system PATH
    ffmpeg_paths = [
        'ffmpeg',  # Try PATH first
        'ffmpeg.exe',  # Explicit .exe
        os.path.join(os.path.dirname(__file__), 'ffmpeg.exe'),  # Script directory
        r'C:\ffmpeg\bin\ffmpeg.exe',  # Common install location
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
    ]

    ffmpeg_major_version = None
    for ffmpeg_path in ffmpeg_paths:
        try:
            result = subprocess.run(
                [ffmpeg_path, '-version'],
                capture_output=True,
                text=True,
                shell=False,
                env=os.environ.copy()  # Explicitly pass environment
            )
            if result.returncode == 0:
                version_match = re.search(r'ffmpeg version (\d+)\.', result.stdout)
                if version_match:
                    ffmpeg_major_version = int(version_match.group(1))
                    print(f"Version {ffmpeg_major_version}")
                    break
        except FileNotFoundError:
            continue

    if ffmpeg_major_version is None:
        print("Could not find or determine FFmpeg version")
        print(f"Checked paths: {ffmpeg_paths}")

except Exception as e:
    print(f"Error getting FFmpeg version: {e}")

# Patch torchaudio.load to avoid torchcodec completely
import soundfile as sf

def patched_load(filepath, *args, **kwargs):
    # Use soundfile directly to avoid torchcodec
    data, samplerate = sf.read(filepath, dtype='float32')
    # Convert to torch tensor and add channel dimension if needed
    audio_tensor = torch.from_numpy(data)
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
    elif audio_tensor.ndim == 2:
        audio_tensor = audio_tensor.T  # Transpose to (channels, samples)
    return audio_tensor, samplerate

torchaudio.load = patched_load

# Your TTS code here
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
tts.tts_to_file(text="This is a test, I am testing now, huzzah", speaker_wav="./test.wav", language="en",
                file_path="output.wav")