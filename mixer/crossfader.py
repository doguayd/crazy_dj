"""
Crossfade and time-stretch between two audio segments.
Kalite öncelikli: native SR korunur, scipy.signal.resample_poly kullanılır.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import resample_poly
from math import gcd
from .effects import apply_out_effects, apply_in_effects


def load_segment(file_path: str, start: float, duration: float) -> tuple[np.ndarray, int]:
    """
    Native SR'de yükler — hiç resample yapmaz.
    Kalite kaybı sıfır.
    """
    with sf.SoundFile(file_path) as f:
        sr = f.samplerate
        start_frame = max(0, int(start * sr))
        frame_count = int(duration * sr)
        f.seek(start_frame)
        data = f.read(frame_count, dtype="float32", always_2d=True)
    return data, sr


def _resample(data: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """
    Yüksek kaliteli resample — scipy.signal.resample_poly (polyphase filter).
    Lineer interpolasyona kıyasla çok daha iyi frekans yanıtı.
    """
    if src_sr == dst_sr:
        return data
    g = gcd(src_sr, dst_sr)
    up, down = dst_sr // g, src_sr // g
    resampled = resample_poly(data, up, down, axis=0, padtype="line")
    return resampled.astype(np.float32)


def crossfade(seg_a: np.ndarray, seg_b: np.ndarray, fade_samples: int) -> np.ndarray:
    """Equal-power crossfade — lineer'e göre daha doğal geçiş."""
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
    Tail of track A → efekt + crossfade → head of track B.
    Native SR korunur — kalite kaybı yok.
    """
    seg_a, sr_a = load_segment(file_a, start=max(0, cue_a - fade_sec), duration=fade_sec * 2)
    seg_b, sr_b = load_segment(file_b, start=cue_b, duration=fade_sec * 2)

    # Çıkış SR: ikisinden yüksek olanı al (FLAC kalitesi korunur)
    out_sr = max(sr_a, sr_b)

    # Gerekirse resample — yüksek kaliteli polyphase filter ile
    if sr_a != out_sr:
        seg_a = _resample(seg_a, sr_a, out_sr)
    if sr_b != out_sr:
        seg_b = _resample(seg_b, sr_b, out_sr)

    # Kanal sayısını eşitle
    channels = min(seg_a.shape[1], seg_b.shape[1])
    seg_a = seg_a[:, :channels]
    seg_b = seg_b[:, :channels]

    # Efekt uygula
    seg_a = apply_out_effects(seg_a, out_sr, effect)
    seg_b = apply_in_effects(seg_b, out_sr, effect)

    # Crossfade
    fade_samples = int(fade_sec * out_sr)
    mixed = crossfade(seg_a, seg_b, fade_samples)

    # Peak normalize — clipping önle, orijinal dinaminği koru
    peak = np.max(np.abs(mixed))
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # FLAC: subtype="PCM_24" ile 24-bit kalite
    sf.write(str(out), mixed, out_sr, subtype="PCM_24")
    return str(out)
