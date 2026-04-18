"""
Song similarity and sync-point detection between two analyzed tracks.
"""

import numpy as np
from .audio_analyzer import SongAnalysis


CAMELOT_WHEEL = {
    "A major": "11B",  "A minor": "8A",
    "A# major": "6B",  "A# minor": "3A",
    "B major": "1B",   "B minor": "10A",
    "C major": "8B",   "C minor": "5A",
    "C# major": "3B",  "C# minor": "12A",
    "D major": "10B",  "D minor": "7A",
    "D# major": "5B",  "D# minor": "2A",
    "E major": "12B",  "E minor": "9A",
    "F major": "7B",   "F minor": "4A",
    "F# major": "2B",  "F# minor": "11A",
    "G major": "9B",   "G minor": "6A",
    "G# major": "4B",  "G# minor": "1A",
}


def camelot_compatible(key_a: str, key_b: str) -> bool:
    """True if two keys are harmonically compatible (Camelot wheel neighbors)."""
    ca = CAMELOT_WHEEL.get(key_a, "")
    cb = CAMELOT_WHEEL.get(key_b, "")
    if not ca or not cb:
        return False
    if ca == cb:
        return True
    num_a, mode_a = int(ca[:-1]), ca[-1]
    num_b, mode_b = int(cb[:-1]), cb[-1]
    if mode_a == mode_b and abs(num_a - num_b) in (0, 1, 11):
        return True
    if mode_a != mode_b and num_a == num_b:
        return True
    return False


def chroma_similarity(a: SongAnalysis, b: SongAnalysis) -> float:
    """Cosine similarity between chroma vectors (0–1)."""
    va = np.array(a.chroma)
    vb = np.array(b.chroma)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def bpm_distance(a: SongAnalysis, b: SongAnalysis) -> float:
    """BPM difference considering half/double time."""
    ratio = a.bpm / b.bpm if b.bpm > 0 else 1.0
    for factor in (1.0, 2.0, 0.5):
        if abs(ratio - factor) < 0.1:
            return abs(a.bpm - b.bpm * factor)
    return abs(a.bpm - b.bpm)


def find_sync_points(a: SongAnalysis, b: SongAnalysis, max_points: int = 5) -> list[dict]:
    """
    Find beat-aligned transition points where both tracks share similar energy.
    Returns a list of dicts: {time_a, time_b, energy_diff, score}
    """
    sync_points = []
    energy_diff = abs(a.energy - b.energy)

    # Sample candidate cue points at 25%, 50%, 75% of each track
    candidates_a = [a.duration * r for r in (0.25, 0.5, 0.75)]
    candidates_b = [b.duration * r for r in (0.25, 0.5, 0.75)]

    def nearest_beat(beat_times: list, target: float) -> float:
        if not beat_times:
            return target
        arr = np.array(beat_times)
        return float(arr[np.argmin(np.abs(arr - target))])

    for ta in candidates_a:
        for tb in candidates_b:
            beat_a = nearest_beat(a.beat_times, ta)
            beat_b = nearest_beat(b.beat_times, tb)
            score = 1.0 - min(energy_diff, 1.0)
            sync_points.append({
                "time_a": round(beat_a, 3),
                "time_b": round(beat_b, 3),
                "energy_diff": round(energy_diff, 4),
                "score": round(score, 4),
            })

    sync_points.sort(key=lambda x: x["score"], reverse=True)
    return sync_points[:max_points]


def compare(a: SongAnalysis, b: SongAnalysis) -> dict:
    return {
        "chroma_similarity": round(chroma_similarity(a, b), 4),
        "bpm_distance": round(bpm_distance(a, b), 2),
        "key_compatible": camelot_compatible(a.key, b.key),
        "sync_points": find_sync_points(a, b),
    }
