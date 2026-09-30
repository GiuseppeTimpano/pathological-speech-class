"""Log-Mel spectrogram front-end of the CNN-BiGRU model."""

from __future__ import annotations

import torch
import torchaudio


def stft_params(sampling_rate: int) -> tuple[int, int, int]:
    """(n_fft, win_length, hop_length) adapted to the sampling rate.

    The hop length scales with the sampling rate, so a 10 s clip always yields 1001 frames.
    """
    if sampling_rate <= 8000:
        return 512, 200, 80
    if sampling_rate <= 16000:
        return 1024, 400, 160
    return 2048, 1102, 441


def log_mel_spectrogram(waveform: torch.Tensor, sampling_rate: int, max_duration: float = 10.0,
                        n_mels: int = 128) -> torch.Tensor:
    """Waveform (T,) -> standardised log-Mel spectrogram of shape (frames, n_mels).

    The clip is truncated or zero-padded to ``max_duration`` seconds, converted to a power
    Mel spectrogram at its native sampling rate, log-compressed and z-normalised per clip.
    """
    max_samples = int(sampling_rate * max_duration)
    waveform = waveform.reshape(1, -1)[:, :max_samples]
    waveform = torch.nn.functional.pad(waveform, (0, max(0, max_samples - waveform.shape[1])))

    n_fft, win_length, hop_length = stft_params(sampling_rate)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        f_min=0.0,
        f_max=sampling_rate // 2,
        n_mels=n_mels,
        power=2.0,
    )(waveform)

    log_mel = torch.log10(mel + 1e-6)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return log_mel.squeeze(0).transpose(0, 1)  # (frames, n_mels)
