"""
Core audio analysis: BPM, key, beat grid, energy, chroma features.
Results are stored in SQLite for Dotify integration.
"""

import librosa
import numpy as np
import soundfile as sf
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class SongAnalysis:
    path: str
    bpm: float
    key: str          # e.g. "C major", "A minor"
    energy: float     # 0.0 - 1.0 RMS normalized
    beat_times: list  # seconds of each beat
    chroma: list      # mean chroma vector (12 values)
    duration: float


def detect_key(chroma_mean: np.ndarray) -> str:
    """Krumhansl-Schmuckler key-finding algorithm."""
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']

    major_scores = [np.corrcoef(np.roll(major_profile, i), chroma_mean)[0, 1]
                    for i in range(12)]
    minor_scores = [np.corrcoef(np.roll(minor_profile, i), chroma_mean)[0, 1]
                    for i in range(12)]

    best_major = int(np.argmax(major_scores))
    best_minor = int(np.argmax(minor_scores))

    if major_scores[best_major] >= minor_scores[best_minor]:
        return f"{note_names[best_major]} major"
    else:
        return f"{note_names[best_minor]} minor"


def analyze(file_path: str) -> SongAnalysis:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    y, sr = librosa.load(str(path), sr=None, mono=True)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    key = detect_key(chroma_mean)

    rms = librosa.feature.rms(y=y)[0]
    energy = float(np.mean(rms) / (np.max(rms) + 1e-9))

    return SongAnalysis(
        path=str(path.resolve()),
        bpm=float(round(float(tempo), 2)),
        key=key,
        energy=round(energy, 4),
        beat_times=beat_times,
        chroma=chroma_mean.tolist(),
        duration=round(float(len(y) / sr), 2),
    )
