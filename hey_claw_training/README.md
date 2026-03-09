# hey_claw - Custom Wake Word Training

A custom openWakeWord model trained to detect the phrase **"hey claw"**, built on macOS arm64 without any Linux-specific dependencies.

## Model

The trained model is at `output/hey_claw.onnx` (plus `output/hey_claw.onnx.data`).

### Using the model

```python
from openwakeword.model import Model

model = Model(wakeword_models=["hey_claw_training/output/hey_claw.onnx"])

# Feed 16-bit 16 kHz PCM audio frames (multiples of 80 ms work best)
prediction = model.predict(audio_frame)
print(prediction)  # {'hey_claw': 0.XX}
```

Or test live from the microphone:

```bash
.venv/bin/python ../examples/detect_from_microphone.py \
  --model_path output/hey_claw.onnx
```

## How it was trained

### Overview

openWakeWord's automated training pipeline normally relies on
[piper-sample-generator](https://github.com/dscripka/piper-sample-generator), which requires
`piper-phonemize` - a package with Linux-only binary wheels. To support macOS arm64,
this repo uses the `piper-tts` Python package instead (which has native arm64 wheels)
to generate synthetic training clips.

Training uses 100% synthetic speech - no real recordings needed.

### Architecture

- **Backbone**: frozen openWakeWord melspectrogram + embedding models (shared feature extractor)
- **Wake word head**: small fully-connected DNN (32 units, ~200K parameters)
- **Input**: 16 x 96 openWakeWord embedding features (~1.28 s of audio)

### Training data

| Split | Phrase type | Count |
|-------|-------------|-------|
| Train positive | "hey claw" | 5,000 clips |
| Train negative | adversarial phrases (phonemically similar) | 5,000 clips |
| Val positive | "hey claw" | 1,000 clips |
| Val negative | adversarial phrases | 1,000 clips |
| Background negative | ACAV100M pre-computed features | ~1,400 hrs |
| Background augmentation | FMA music | 1 hr |
| Room impulse responses | MIT IR Survey | 270 files |
| False-positive validation | openWakeWord validation set | ~11 hrs |

Clip generation used the **libritts_r-medium** piper voice model (904 speakers) with
randomized `noise_scale`, `noise_w_scale`, and `length_scale` for variety.

Adversarial negatives were generated automatically from phoneme overlap with "hey claw"
using `openwakeword.data.generate_adversarial_texts`.

### Steps to reproduce

#### 1. Environment

```bash
cd hey_claw_training
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch torchaudio torchinfo torchmetrics \
    speechbrain==0.5.14 audiomentations torch-audiomentations \
    mutagen pyyaml "scipy<1.13" tqdm numpy "datasets==2.14.6" \
    "pyarrow<14" webrtcvad pronouncing acoustics "setuptools<71" \
    piper-tts soundfile onnxscript
uv pip install -e ..
```

#### 2. Download piper voice model

```bash
curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx" \
  -o libritts_r-medium.onnx
curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json" \
  -o libritts_r-medium.onnx.json
```

#### 3. Download datasets

```bash
# MIT Room Impulse Responses (~6 MB, 270 files)
.venv/bin/python - <<'EOF'
import datasets, scipy.io.wavfile, numpy as np, os
from tqdm import tqdm
os.makedirs("mit_rirs", exist_ok=True)
ds = datasets.load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True)
for row in tqdm(ds):
    name = row["audio"]["path"].split("/")[-1]
    scipy.io.wavfile.write(f"mit_rirs/{name}", 16000, (row["audio"]["array"]*32767).astype(np.int16))
EOF

# FMA music background (~110 MB, 1 hour)
.venv/bin/python - <<'EOF'
import datasets, scipy.io.wavfile, numpy as np, os
from tqdm import tqdm
os.makedirs("fma", exist_ok=True)
ds = datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True)
ds = iter(ds.cast_column("audio", datasets.Audio(sampling_rate=16000)))
for i in tqdm(range(120)):
    row = next(ds)
    name = row["audio"]["path"].split("/")[-1].replace(".mp3", ".wav")
    scipy.io.wavfile.write(f"fma/{name}", 16000, (row["audio"]["array"]*32767).astype(np.int16))
EOF

# Pre-computed openWakeWord features (~16 GB - needs ~17 GB free disk)
curl -L "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy" \
  -o openwakeword_features_ACAV100M_2000_hrs_16bit.npy

# Validation set features (~176 MB)
curl -L "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy" \
  -o validation_set_features.npy
```

> **Disk space note:** If you run out of disk space during the ACAV100M download, run
> `python fix_truncated_npy.py openwakeword_features_ACAV100M_2000_hrs_16bit.npy` to repair
> the partial file and use however many rows were downloaded.

#### 4. Download openWakeWord backbone models

```bash
mkdir -p ../openwakeword/resources/models
curl -L "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx" \
  -o ../openwakeword/resources/models/melspectrogram.onnx
curl -L "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx" \
  -o ../openwakeword/resources/models/embedding_model.onnx
```

#### 5. Generate synthetic clips

```bash
.venv/bin/python generate_clips_piper.py --training_config hey_claw.yml
```

#### 6. Augment clips and compute features

```bash
.venv/bin/python -m openwakeword.train --training_config hey_claw.yml --augment_clips
```

#### 7. Train the model

```bash
.venv/bin/python -m openwakeword.train --training_config hey_claw.yml --train_model
```

Output: `output/hey_claw.onnx` + `output/hey_claw.onnx.data`

### Patches to train.py

Three small fixes were needed to run on macOS arm64 (see `openwakeword/train.py` diff):

1. **Conditional piper import** - `generate_samples` import moved inside `if args.generate_clips is True:` block so `--augment_clips` and `--train_model` don't require piper-sample-generator.

2. **`args.generate_clips is True`** - upstream uses `default="False"` (a string) in argparse, which is truthy; changed comparison from `if args.generate_clips` to `if args.generate_clips is True`.

3. **`num_workers=0` in DataLoader** - macOS uses `spawn` for multiprocessing; lambdas in the batch generator can't be pickled across processes, so workers are disabled.

### Patch to torch_audiomentations

`torchaudio.info()` was removed in torchaudio 2.x. A compatibility shim using `soundfile`
is injected at the top of `.venv/lib/python3.11/site-packages/torch_audiomentations/utils/io.py`:

```python
if not hasattr(torchaudio, "info"):
    import soundfile as sf

    class _AudioInfo:
        def __init__(self, num_frames, sample_rate):
            self.num_frames = num_frames
            self.sample_rate = sample_rate

    def _info(path):
        info = sf.info(path)
        return _AudioInfo(info.frames, info.samplerate)

    torchaudio.info = _info
```

## Files in this directory

| File | Purpose |
|------|---------|
| `hey_claw.yml` | Training configuration |
| `generate_clips_piper.py` | Synthetic clip generator (uses piper-tts, arm64 compatible) |
| `fix_truncated_npy.py` | Repairs a partially-downloaded .npy file |
| `output/hey_claw.onnx` | Trained wake word model |
| `output/hey_claw.onnx.data` | External weights for the ONNX model |
| `libritts_r-medium.onnx*` | Piper TTS voice model (not committed - download separately) |
