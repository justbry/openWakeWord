"""
Generate synthetic training clips for "hey claw" using the piper-tts Python package (arm64 Mac compatible).
Replaces piper-sample-generator which requires piper-phonemize (Linux-only wheels).
"""

import os
import sys
import uuid
import random
import scipy.io.wavfile
import scipy.signal
import numpy as np
from pathlib import Path
from tqdm import tqdm

from piper.voice import PiperVoice
from piper.config import SynthesisConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
from openwakeword.data import generate_adversarial_texts

MODEL      = str(Path(__file__).parent / "libritts_r-medium.onnx")
MODEL_JSON = str(Path(__file__).parent / "libritts_r-medium.onnx.json")
N_SPEAKERS = 904   # libritts_r-medium has 904 speakers
TARGET_SR  = 16000 # openWakeWord requires 16 kHz

# Synthesis parameter variations for diversity
NOISE_SCALES   = [0.5, 0.667, 0.85, 0.98]
NOISE_W_SCALES = [0.6, 0.8, 0.95]
LENGTH_SCALES  = [0.75, 0.9, 1.0, 1.1, 1.25]

print("Loading piper voice model...")
VOICE = PiperVoice.load(MODEL, config_path=MODEL_JSON, use_cuda=False)
SOURCE_SR = 22050  # piper libritts model outputs 22050 Hz


def generate_clip(text: str, output_path: str):
    syn_config = SynthesisConfig(
        speaker_id=random.randint(0, N_SPEAKERS - 1),
        noise_scale=random.choice(NOISE_SCALES),
        noise_w_scale=random.choice(NOISE_W_SCALES),
        length_scale=random.choice(LENGTH_SCALES),
    )
    try:
        chunks = list(VOICE.synthesize(text, syn_config))
        if not chunks:
            return False
        audio = np.concatenate([c.audio_float_array for c in chunks])
        audio_16k = scipy.signal.resample_poly(audio, TARGET_SR, SOURCE_SR)
        audio_int16 = (audio_16k * 32767).astype(np.int16)
        scipy.io.wavfile.write(output_path, TARGET_SR, audio_int16)
        return True
    except Exception:
        return False


def generate_clips(phrases, output_dir, n_samples, desc=""):
    os.makedirs(output_dir, exist_ok=True)
    existing = len(list(Path(output_dir).glob("*.wav")))
    needed = n_samples - existing
    if needed <= 0:
        print(f"  Skipping {desc}: already have {existing} >= {n_samples} clips")
        return

    print(f"  Generating {needed} clips for: {desc}")
    generated = 0
    pbar = tqdm(total=needed)

    while generated < needed:
        phrase = random.choice(phrases)
        out_path = os.path.join(output_dir, uuid.uuid4().hex + ".wav")
        if generate_clip(phrase, out_path):
            generated += 1
            pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.training_config))
    out = config["output_dir"]
    model_name = config["model_name"]
    n_samples = config["n_samples"]
    n_val = config["n_samples_val"]
    target_phrases = config["target_phrase"]

    base = os.path.join(out, model_name)
    os.makedirs(base, exist_ok=True)

    # Build adversarial phrase list
    print("\nBuilding adversarial phrase list...")
    adversarial = list(config.get("custom_negative_phrases", []))
    for phrase in target_phrases:
        adversarial.extend(
            generate_adversarial_texts(
                input_text=phrase,
                N=n_samples // len(target_phrases),
                include_partial_phrase=1.0,
                include_input_words=0.2,
            )
        )
    adversarial = list(set(adversarial))
    print(f"  {len(adversarial)} adversarial phrases")

    # Generate all four clip sets
    generate_clips(target_phrases, os.path.join(base, "positive_train"), n_samples, "positive train")
    generate_clips(target_phrases, os.path.join(base, "positive_test"),  n_val,     "positive val")
    generate_clips(adversarial,    os.path.join(base, "negative_train"), n_samples, "negative train")
    generate_clips(adversarial,    os.path.join(base, "negative_test"),  n_val,     "negative val")

    print("\nClip generation complete!")
