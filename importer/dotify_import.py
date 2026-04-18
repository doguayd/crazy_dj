"""
Dotify kütüphanesini crazy_dj'e import eder.
Dotify.db3'teki Songs tablosunu okur, FilePath'leri alır ve analiz eder.
Opsiyonel: analiz sonuçlarını (BPM, Key, Energy) Dotify.db3'e geri yazar.
"""

import sqlite3
import json
import glob
import os
from pathlib import Path


# MAUI AppDataDirectory on Windows:
# C:\Users\{user}\AppData\Local\Packages\{pkg}\LocalState\Dotify.db3
def find_dotify_db() -> str | None:
    pattern = os.path.expanduser(
        r"~\AppData\Local\Packages\*Dotify*\LocalState\Dotify.db3"
    )
    matches = glob.glob(pattern)
    if matches:
        return matches[0]

    # Fallback: debug/sideload path
    fallback = os.path.expanduser(r"~\AppData\Local\Dotify\Dotify.db3")
    if os.path.exists(fallback):
        return fallback

    return None


def read_songs(db_path: str) -> list[dict]:
    """Dotify.db3'teki tüm indirilen şarkıları döndürür."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT Id, Title, Artist, AlbumName, FilePath, BPM, Energy, [Key], Tempo
        FROM Songs
        WHERE IsDownloaded = 1 AND FilePath != ''
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def write_analysis_back(db_path: str, song_id: str, bpm: float, energy: float, key_index: int):
    """
    Analiz sonuçlarını Dotify.db3'e yazar.
    key_index: Spotify'ın Pitch Class (0=C, 1=C#, ..., 11=B) — -1 bilinmiyorsa
    """
    conn = sqlite3.connect(db_path)
    conn.execute("""
        UPDATE Songs
        SET BPM = ?, Tempo = ?, Energy = ?, [Key] = ?
        WHERE Id = ?
    """, (bpm, bpm, energy, key_index, song_id))
    conn.commit()
    conn.close()


KEY_NAME_TO_PITCH_CLASS = {
    "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
    "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11,
}

def key_str_to_pitch_class(key_str: str) -> int:
    """'A minor' veya 'C major' → Spotify pitch class int."""
    root = key_str.split()[0]
    return KEY_NAME_TO_PITCH_CLASS.get(root, -1)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from analyzer.audio_analyzer import analyze
    from db import store

    store.init_db()

    db_path = find_dotify_db()
    if not db_path:
        # Manuel yol
        db_path = input("Dotify.db3 yolunu girin: ").strip().strip('"')

    print(f"Dotify DB: {db_path}")
    songs = read_songs(db_path)
    print(f"{len(songs)} indirilen şarkı bulundu.\n")

    write_back = "--write-back" in sys.argv

    for song in songs:
        path = song["FilePath"]
        if not os.path.exists(path):
            print(f"  [SKIP] Dosya yok: {path}")
            continue

        cached = store.load(path)
        if cached:
            print(f"  [cached] {song['Title']} — {cached.bpm:.1f} BPM, {cached.key}")
            continue

        print(f"  Analyzing: {song['Title']} ...", end=" ", flush=True)
        try:
            result = analyze(path)
            store.save(result)
            print(f"{result.bpm:.1f} BPM, {result.key}")

            if write_back:
                pitch = key_str_to_pitch_class(result.key)
                write_analysis_back(db_path, song["Id"], result.bpm, result.energy, pitch)
                print(f"    → Dotify.db3 güncellendi")
        except Exception as e:
            print(f"HATA: {e}")
