"""
Downbeat tespiti — beat-this (varsa) veya librosa fallback.

beat-this: PyTorch tabanlı modern beat/downbeat dedektörü.
  https://github.com/CPJKU/beat-this  (Python 3.10+, aktif geliştirme)

madmom Python 3.10+ desteklemediği için kullanılmıyor.
"""

import numpy as np


def _detect_with_beat_this(file_path: str) -> list[float]:
    """beat-this ile downbeat zamanları."""
    from beat_this.inference import File2Beats
    predictor = File2Beats(device="cpu", dbn=True)
    beats, downbeats = predictor(file_path)
    return list(downbeats)


def _detect_with_librosa(file_path: str) -> list[float]:
    """librosa fallback — tüm beatler."""
    import librosa
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    return librosa.frames_to_time(beat_frames, sr=sr).tolist()


def get_downbeats(file_path: str) -> tuple[list[float], str]:
    """
    Downbeat zamanlarını döndürür.
    Returns: (beat_times, method)
    """
    try:
        beats = _detect_with_beat_this(file_path)
        if beats:
            return beats, "beat-this"
    except ImportError:
        pass
    except Exception as e:
        print(f"  [downbeat] beat-this hatası: {e}, librosa'ya geçiliyor")

    return _detect_with_librosa(file_path), "librosa"


def best_cue_point(beat_times: list[float], duration: float, prefer_ratio: float = 0.75) -> float:
    """Şarkının %75'ine en yakın downbeat'i cue noktası olarak seç."""
    if not beat_times:
        return duration * prefer_ratio
    target = duration * prefer_ratio
    arr = np.array(beat_times)
    return float(arr[np.argmin(np.abs(arr - target))])
