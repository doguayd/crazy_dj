"""
Downbeat tespiti — madmom (varsa) veya librosa fallback.

madmom, RNN tabanlı downbeat dedektörü ile çok daha hassas
beat/cue noktaları tespit eder. Birden fazla projede standart:
- Auto-DJ (ddman1101), dnb-autodj-3, Automix hepsi madmom kullanır.
"""

import numpy as np
from pathlib import Path


def _detect_with_madmom(file_path: str) -> list[float]:
    """madmom RNNDownBeatProcessor ile downbeat zamanlarını döndürür."""
    import madmom
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor

    proc = RNNDownBeatProcessor()(file_path)
    tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    beats = tracker(proc)
    # beats: [[time, beat_number], ...] — sadece downbeat (beat_number==1) al
    downbeats = [b[0] for b in beats if int(b[1]) == 1]
    return downbeats


def _detect_with_librosa(file_path: str) -> list[float]:
    """librosa fallback — tüm beatler (downbeat değil)."""
    import librosa
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    return librosa.frames_to_time(beat_frames, sr=sr).tolist()


def get_downbeats(file_path: str) -> tuple[list[float], str]:
    """
    Downbeat zamanlarını döndürür.
    Returns: (beat_times, method) — method: "madmom" veya "librosa"
    """
    try:
        beats = _detect_with_madmom(file_path)
        if beats:
            return beats, "madmom"
    except ImportError:
        pass
    except Exception as e:
        print(f"  [downbeat] madmom hatası: {e}, librosa'ya geçiliyor")

    return _detect_with_librosa(file_path), "librosa"


def best_cue_point(
    beat_times: list[float],
    duration: float,
    prefer_ratio: float = 0.75,
) -> float:
    """
    Şarkının %75'ine en yakın downbeat'i cue noktası olarak seç.
    DJ pratiğinde geçiş genellikle şarkının son çeyreğinde olur.
    """
    if not beat_times:
        return duration * prefer_ratio
    target = duration * prefer_ratio
    arr = np.array(beat_times)
    return float(arr[np.argmin(np.abs(arr - target))])
