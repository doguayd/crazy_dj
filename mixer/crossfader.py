"""
Crossfade and time-stretch between two audio segments.
Uses soundfile + numpy; no real-time dependency.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from .effects import apply_out_effects, apply_in_effects


TARGET_SR = 22050


def load_segment(file_path: str, start: float, duration: float, target_sr: int = TARGET_SR):
    """Load a time slice from an audio file."""
    with sf.SoundFile(file_path) as f:
        sr = f.samplerate
        start_frame = int(start * sr)
        frame_count = int(duration * sr)
        f.seek(start_frame)
        data = f.read(frame_count, dtype="float32", always_2d=True)
    if sr != target_sr:
        factor = target_sr / sr
        new_len = int(len(data) * factor)
        data = np.array([np.interp(
            np.linspace(0, len(data) - 1, new_len), np.arange(len(data)), data[:, ch]
        ) for ch in range(data.shape[1])]).T
    return data.astype(np.float32), target_sr


def crossfade(seg_a: np.ndarray, seg_b: np.ndarray, fade_samples: int) -> np.ndarray:
    """Equal-power crossfade (daha doğal geçiş, linear'dan üstün)."""
    fade_samples = min(fade_samples, len(seg_a), len(seg_b))
    t = np.linspace(0.0, np.pi / 2, fade_samples)
    ramp_out = np.cos(t)[:, np.newaxis]
    ramp_in  = np.sin(t)[:, np.newaxis]

    pre   = seg_a[:-fade_samples]
    cross = seg_a[-fade_samples:] * ramp_out + seg_b[:fade_samples] * ramp_in
    post  = seg_b[fade_samples:]

    return np.concatenate([pre, cross, post], axis=0)


def mix_transition(
    file_a: str, cue_a: float,
    file_b: str, cue_b: float,
    fade_sec: float = 8.0,
    output_path: str = "output/transition.flac",
    effect: str = "sweep",
) -> str:
    """
    Tail of track A → crossfade → head of track B, with effects.
    effect presets: clean | sweep | echo | deep | energetic
    """
    seg_a, sr = load_segment(file_a, start=max(0, cue_a - fade_sec), duration=fade_sec * 2)
    seg_b, _  = load_segment(file_b, start=cue_b, duration=fade_sec * 2)

    channels = min(seg_a.shape[1], seg_b.shape[1])
    seg_a = seg_a[:, :channels]
    seg_b = seg_b[:, :channels]

    # Efekt uygula
    seg_a = apply_out_effects(seg_a, sr, effect)
    seg_b = apply_in_effects(seg_b, sr, effect)

    fade_samples = int(fade_sec * sr)
    mixed = crossfade(seg_a, seg_b, fade_samples)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), mixed, sr)
    return str(out)
