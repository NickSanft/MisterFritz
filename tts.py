import hashlib
import sys
import logging
import argparse
import threading
import torch
from pathlib import Path
from typing import Optional, List
from TTS.api import TTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TTSEngine:
    """
    A wrapper class for Coqui TTS to handle text-to-speech generation
    with caching, dynamic speaker selection, and GPU acceleration.
    """

    # Default constants
    DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
    DEFAULT_OUTPUT_DIR = Path("output")

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        """
        Initialize the TTS engine.

        Args:
            model_name (str): The specific TTS model to load.
            device (str): 'cuda' or 'cpu'. Auto-detects if None.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = self.DEFAULT_OUTPUT_DIR

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Coqui's TTS object is not safe for concurrent inference on one
        # instance. Callers now reach generate_speech from a worker-thread
        # pool, so serialise here rather than trusting every caller to.
        self._synth_lock = threading.Lock()

        logger.info(f"Initializing TTS on device: {self.device}")
        try:
            self.tts = TTS(self.model_name).to(self.device)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def _get_hex_hash(self, text: str, speaker_id: str) -> str:
        """
        Generate a unique hash based on text AND speaker.
        This prevents overwriting if the same text is said by different voices.
        """
        unique_string = f"{text}_{speaker_id}"
        return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

    def get_available_speakers(self) -> List[str]:
        """Return a list of available speakers depending on the model loaded."""
        if hasattr(self.tts, 'speakers') and self.tts.speakers:
            return self.tts.speakers
        return []

    def generate_speech(self,
                        message: str,
                        speaker: str = "Baldur Sanjin",
                        language: str = "en",
                        reference_wav: Optional[str] = None,
                        force_regenerate: bool = False) -> str:
        """
        Generate audio from text.

        Synchronous. This was declared `async def` while containing no `await`
        at all, which meant awaiting it ran XTTS inference inline on the caller's
        event loop — and left the CLI at the bottom of this file printing a
        coroutine object instead of synthesising anything. Call it from a worker
        thread (see bot_adapters.run_blocking).

        Args:
            message (str): Text to speak.
            speaker (str): Name of the speaker (if using multi-speaker model).
            language (str): Language code (en, es, fr, etc.).
            reference_wav (str): Path to a wav file for voice cloning (overrides speaker param).
            force_regenerate (bool): If True, ignores cache and creates new file.

        Returns:
            str: Path to the generated .wav file.
        """
        # Create a clean filename
        file_hash = self._get_hex_hash(message, reference_wav if reference_wav else speaker)
        output_path = self.output_dir / f"speech_{file_hash}.wav"

        # Check Cache
        if output_path.exists() and not force_regenerate:
            logger.info(f"Cached file found: {output_path}")
            return str(output_path)

        logger.info(f"Generating audio for: '{message[:30]}...'")

        try:
            # XTTS specific logic for voice cloning vs standard speaker
            with self._synth_lock:
                if reference_wav:
                    logger.info(f"Cloning voice from: {reference_wav}")
                    self.tts.tts_to_file(
                        text=message,
                        file_path=str(output_path),
                        speaker_wav=reference_wav,
                        language=language,
                        split_sentences=True
                    )
                else:
                    logger.info(f"Using speaker: {speaker}")
                    self.tts.tts_to_file(
                        text=message,
                        file_path=str(output_path),
                        speaker=speaker,
                        language=language,
                        split_sentences=True
                    )

            logger.info(f"Generation successful: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            raise


def main():
    """CLI Entry point for quick testing."""
    parser = argparse.ArgumentParser(description="Generate Speech from Text")
    parser.add_argument("text", type=str, help="The text to convert to speech")
    parser.add_argument("--speaker", type=str, default="Baldur Sanjin", help="Speaker name")
    parser.add_argument("--ref", type=str, help="Path to reference WAV for cloning")
    parser.add_argument("--lang", type=str, default="en", help="Language code")
    parser.add_argument("--list-speakers", action="store_true", help="List available speakers and exit")

    args = parser.parse_args()

    # Initialize Engine
    engine = TTSEngine()

    if args.list_speakers:
        print("Available Speakers:", engine.get_available_speakers())
        sys.exit(0)

    # Generate
    try:
        output = engine.generate_speech(
            message=args.text,
            speaker=args.speaker,
            language=args.lang,
            reference_wav=args.ref
        )
        print(f"File generated at: {output}")
    except Exception as e:
        print(f"Failed: {e}")


if __name__ == "__main__":
    main()
