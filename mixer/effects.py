"""
Geçiş efektleri: low-pass sweep, high-pass sweep, reverb tail, echo.
Tüm efektler numpy array üzerinde çalışır (soundfile uyumlu).
"""

import numpy as np
from scipy.signal import butter, sosfilt


# ── Filtreler ─────────────────────────────────────────────────────────────────

def _butter_filter(data: np.ndarray, sr: int, cutoff: float, btype: str) -> np.ndarray:
    nyq = sr / 2
    cutoff = np.clip(cutoff, 20, nyq - 1)
    sos = butter(4, cutoff / nyq, btype=btype, output="sos")
    return sosfilt(sos, data, axis=0).astype(np.float32)


def lowpass(data: np.ndarray, sr: int, cutoff_hz: float = 800.0) -> np.ndarray:
    return _butter_filter(data, sr, cutoff_hz, "low")


def highpass(data: np.ndarray, sr: int, cutoff_hz: float = 800.0) -> np.ndarray:
    return _butter_filter(data, sr, cutoff_hz, "high")


# ── Sweep efektleri ───────────────────────────────────────────────────────────

def lowpass_sweep(
    data: np.ndarray, sr: int,
    start_hz: float = 8000.0, end_hz: float = 200.0
) -> np.ndarray:
    """Crossfade sırasında low-pass filtre giderek kapanır (basın sesi kalır)."""
    n = len(data)
    result = np.zeros_like(data)
    steps = 32
    chunk = n // steps
    for i in range(steps):
        cutoff = start_hz + (end_hz - start_hz) * (i / steps)
        sl = slice(i * chunk, (i + 1) * chunk if i < steps - 1 else n)
        result[sl] = _butter_filter(data[sl], sr, max(cutoff, 30), "low")
    return result


def highpass_sweep(
    data: np.ndarray, sr: int,
    start_hz: float = 200.0, end_hz: float = 8000.0
) -> np.ndarray:
    """Giren şarkı giderek açılır (bas önce gelir, tiz sonra)."""
    n = len(data)
    result = np.zeros_like(data)
    steps = 32
    chunk = n // steps
    for i in range(steps):
        cutoff = start_hz + (end_hz - start_hz) * (i / steps)
        sl = slice(i * chunk, (i + 1) * chunk if i < steps - 1 else n)
        result[sl] = _butter_filter(data[sl], sr, min(cutoff, sr / 2 - 1), "high")
    return result


# ── Reverb / Echo ─────────────────────────────────────────────────────────────

def reverb_tail(
    data: np.ndarray, sr: int,
    decay: float = 0.4, delay_ms: float = 60.0
) -> np.ndarray:
    """Basit comb-filter reverb — çıkış şarkısının sonuna derinlik katar."""
    delay_samples = int(sr * delay_ms / 1000)
    out = data.copy()
    for i in range(delay_samples, len(out)):
        out[i] += out[i - delay_samples] * decay
    # Normalize — clipping engelle
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out /= peak
    return out.astype(np.float32)


def echo(
    data: np.ndarray, sr: int,
    delay_ms: float = 375.0, feedback: float = 0.35, mix: float = 0.4
) -> np.ndarray:
    """Tek yankı echo — 375ms default (tipik 160 BPM quarter note)."""
    delay_samples = int(sr * delay_ms / 1000)
    wet = np.zeros_like(data)
    for i in range(delay_samples, len(data)):
        wet[i] = data[i - delay_samples] * feedback
    out = data * (1 - mix) + wet * mix
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out /= peak
    return out.astype(np.float32)


# ── Efekt zinciri ─────────────────────────────────────────────────────────────

EFFECT_PRESETS = {
    "clean":    [],
    "sweep":    ["lowpass_sweep_out", "highpass_sweep_in"],
    "echo":     ["echo_out", "reverb_in"],
    "deep":     ["lowpass_sweep_out", "reverb_in", "echo_out"],
    "energetic":["highpass_sweep_in", "echo_out"],
}


def apply_out_effects(data: np.ndarray, sr: int, preset: str) -> np.ndarray:
    effects = EFFECT_PRESETS.get(preset, [])
    if "lowpass_sweep_out" in effects:
        data = lowpass_sweep(data, sr)
    if "echo_out" in effects:
        data = echo(data, sr)
    if "reverb_out" in effects:
        data = reverb_tail(data, sr)
    return data


def apply_in_effects(data: np.ndarray, sr: int, preset: str) -> np.ndarray:
    effects = EFFECT_PRESETS.get(preset, [])
    if "highpass_sweep_in" in effects:
        data = highpass_sweep(data, sr)
    if "reverb_in" in effects:
        data = reverb_tail(data, sr, decay=0.2)
    return data
