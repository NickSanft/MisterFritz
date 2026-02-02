import os
import glob
import sys
import subprocess

# Trainer: Where the training loop happens
from trainer import Trainer, TrainerArgs

# Configs: Define model and dataset structure
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.api import TTS


def train_model():
    # -------------------------------------------------------------------------
    # 1. SETUP PATHS
    # -------------------------------------------------------------------------
    # Directory where you want to save the model checkpoints
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

    # Path to your dataset (Change this to your actual dataset path)
    # The dataset must follow LJSpeech format:
    # /path/to/dataset/metadata.csv and /path/to/dataset/wavs/
    dataset_path = "./Omniman"

    # -------------------------------------------------------------------------
    # 2. CONFIGURE DATASET
    # -------------------------------------------------------------------------
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",  # The format of your metadata (ljspeech is standard)
        meta_file_train="metadata.csv",
        path=dataset_path
    )

    # -------------------------------------------------------------------------
    # 3. CONFIGURE MODEL (GlowTTS)
    # -------------------------------------------------------------------------
    config = GlowTTSConfig(
        batch_size=32,
        eval_batch_size=16,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=100,  # Adjust based on when loss flattens
        text_cleaner="phoneme_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=25,
        print_eval=False,
        mixed_precision=True,  # Set to False if you don't have a GPU
        output_path=output_path,
        datasets=[dataset_config],
        save_step=1000,  # Save backup every 1000 steps
    )

    # -------------------------------------------------------------------------
    # 4. INITIALIZE AUDIO PROCESSOR & TOKENIZER
    # -------------------------------------------------------------------------
    # Audio processor extracts features (spectrograms) from audio
    ap = AudioProcessor.init_from_config(config)

    # Tokenizer converts text to phonemes/tokens
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # -------------------------------------------------------------------------
    # 5. LOAD DATA SAMPLES
    # -------------------------------------------------------------------------
    # This splits the dataset into training and evaluation sets automatically
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=1,
    )

    # -------------------------------------------------------------------------
    # 6. INITIALIZE MODEL
    # -------------------------------------------------------------------------
    model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

    # -------------------------------------------------------------------------
    # 7. INITIALIZE TRAINER
    # -------------------------------------------------------------------------
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    # -------------------------------------------------------------------------
    # 8. START TRAINING
    # -------------------------------------------------------------------------
    print("🚀 Starting training... Check tensorboard for logs.")
    trainer.fit()

    # -------------------------------------------------------------------------
    # 9. GENERATE AND PLAY AUDIO (INFERENCE)
    # -------------------------------------------------------------------------
    print("\n🏁 Training loop finished. Locating best model for inference...")

    # Find the most recent run directory created by the Trainer
    all_runs = glob.glob(os.path.join(output_path, "*"))
    all_runs = [d for d in all_runs if os.path.isdir(d)]

    if not all_runs:
        print("❌ No run directories found in output path.")
        return

    # Get the latest run folder based on creation time
    latest_run_dir = max(all_runs, key=os.path.getctime)
    best_model_path = os.path.join(latest_run_dir, "best_model.pth")
    config_path = os.path.join(latest_run_dir, "config.json")

    if not os.path.exists(best_model_path):
        # Fallback: check for the most recent checkpoint if 'best_model.pth' isn't there yet
        checkpoints = glob.glob(os.path.join(latest_run_dir, "checkpoint_*.pth"))
        if checkpoints:
            best_model_path = max(checkpoints, key=os.path.getctime)
        else:
            print(f"❌ Could not find a model file in {latest_run_dir}")
            return

    print(f"✅ Loading model from: {best_model_path}")

    # Initialize the inference API
    # We use gpu=False here to ensure it doesn't crash if VRAM is still tied up by the trainer
    tts = TTS(model_path=best_model_path, config_path=config_path, progress_bar=False, gpu=False)

    sample_text = "The training has finished successfully. This is a test of the new voice model."
    output_wav = "output_generated.wav"

    print(f"🎙️ Generating audio: '{sample_text}'")
    tts.tts_to_file(text=sample_text, file_path=output_wav)

    print(f"💾 Saved to: {output_wav}")
    print("▶️ Attempting to play audio...")

    # Cross-platform playback logic
    try:
        if sys.platform == "win32":
            os.startfile(output_wav)
        elif sys.platform == "darwin":  # macOS
            subprocess.call(["afplay", output_wav])
        else:  # Linux
            try:
                # Try xdg-open (generic) then aplay (standard ALSA)
                subprocess.call(["xdg-open", output_wav])
            except FileNotFoundError:
                subprocess.call(["aplay", output_wav])
    except Exception as e:
        print(f"Could not play audio automatically: {e}")
        print(f"Please listen to '{output_wav}' manually.")


if __name__ == "__main__":
    train_model()