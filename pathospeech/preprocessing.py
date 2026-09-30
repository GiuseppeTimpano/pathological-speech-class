"""Signal cleaning (voice activity detection + Wiener filter) and dataset index creation."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import wiener
from tqdm import tqdm

from pathospeech.config import CLASS_FOLDERS


def voice_activity_detection(audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Keep only voiced frames.

    A frame is voiced when its short-term energy is above the 70th percentile and both its
    zero-crossing rate and spectral flatness are below the 60th percentile. Voiced frames are
    concatenated; if none is found the original signal is returned unchanged.
    """
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0] ** 2
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=frame_length, hop_length=hop_length)[0]

    voiced = (
        (energy > np.percentile(energy, 70))
        & (zcr < np.percentile(zcr, 60))
        & (flatness < np.percentile(flatness, 60))
    )

    segments = [audio[i * hop_length: i * hop_length + frame_length] for i in np.flatnonzero(voiced)]
    if not segments:
        return audio
    return np.concatenate(segments)


def wiener_filter(signal: np.ndarray) -> np.ndarray:
    """Wiener denoising (scipy defaults); silent or empty signals are returned unchanged."""
    if len(signal) == 0 or np.var(signal) == 0:
        return signal
    return np.nan_to_num(wiener(signal))


def clean_recording(path: Path) -> tuple[np.ndarray, int]:
    """Load a recording at its native sampling rate, apply VAD and Wiener filtering."""
    audio, sr = librosa.load(path, sr=None)
    return wiener_filter(voice_activity_detection(audio)), sr


def build_dataset(raw_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Clean every recording and write ``out_dir/dataset.csv``.

    Expected input layout (one sub-folder per subject)::

        raw_dir/Healthy/<subject_id>/*.wav
        raw_dir/Disease_PD/<subject_id>/*.wav
        raw_dir/Disease_ALS/<subject_id>/*.wav

    The CSV stores paths relative to ``out_dir``, the subject id and the multiclass label
    (0 = HS, 1 = PD, 2 = ALS); the binary label is derived from it at training time.
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    rows = []
    for class_name, label in CLASS_FOLDERS.items():
        class_dir = raw_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for subject_dir in tqdm(sorted(p for p in class_dir.iterdir() if p.is_dir()), desc=class_name):
            for wav in sorted(subject_dir.glob("*.wav")):
                signal, sr = clean_recording(wav)
                rel_path = Path(class_name) / subject_dir.name / wav.name
                (out_dir / rel_path).parent.mkdir(parents=True, exist_ok=True)
                sf.write(out_dir / rel_path, signal, sr)
                rows.append({"filename": rel_path.as_posix(), "label": label, "patient_id": subject_dir.name})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "dataset.csv", index=False)
    return df
