"""
Kütüphanedeki tüm şarkılar arasında uyumlu çiftleri bulur.
Camelot wheel, BPM yakınlığı ve chroma similarity birlikte değerlendirilir.
"""

import numpy as np
from .audio_analyzer import SongAnalysis
from .similarity import camelot_compatible, chroma_similarity, bpm_distance


def compatibility_score(a: SongAnalysis, b: SongAnalysis) -> float:
    """
    0.0 – 1.0 arası uyum skoru.
    Ağırlıklar: key %40, BPM %35, chroma %25
    """
    # Key uyumu
    key_score = 1.0 if camelot_compatible(a.key, b.key) else 0.0

    # BPM skoru — 0 BPM farkı=1.0, 20+ fark=0.0
    bpm_diff = bpm_distance(a, b)
    bpm_score = max(0.0, 1.0 - bpm_diff / 20.0)

    # Chroma benzerliği
    chroma_score = chroma_similarity(a, b)

    return round(0.40 * key_score + 0.35 * bpm_score + 0.25 * chroma_score, 4)


def find_compatible_pairs(
    songs: list[SongAnalysis],
    min_score: float = 0.55,
    top_n: int = 10,
) -> list[dict]:
    """
    Tüm şarkı çiftlerini karşılaştırıp en uyumluları döndürür.
    """
    results = []
    n = len(songs)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = songs[i], songs[j]
            score = compatibility_score(a, b)
            if score >= min_score:
                results.append({
                    "score": score,
                    "song_a": a,
                    "song_b": b,
                    "bpm_a": a.bpm,
                    "bpm_b": b.bpm,
                    "key_a": a.key,
                    "key_b": b.key,
                    "key_compatible": camelot_compatible(a.key, b.key),
                })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


def build_setlist(
    songs: list[SongAnalysis],
    start_song: SongAnalysis | None = None,
    length: int = 8,
) -> list[SongAnalysis]:
    """
    Greedy algoritma ile en uyumlu sıralı setlist oluşturur.
    Her adımda mevcut şarkıya en uyumlu olan bir sonrakine geçer.
    """
    remaining = list(songs)
    if start_song is None:
        # En yüksek enerjili şarkıdan başla
        current = max(remaining, key=lambda s: s.energy)
    else:
        current = start_song

    remaining = [s for s in remaining if s.path != current.path]
    setlist = [current]

    while remaining and len(setlist) < length:
        scored = [(compatibility_score(current, s), s) for s in remaining]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best = scored[0]
        if best_score < 0.3:
            break
        setlist.append(best)
        remaining.remove(best)
        current = best

    return setlist
