"""
CLAP (Contrastive Language-Audio Pretraining) tabanlı semantik arama.
AI-DJ-Software'den ilham alındı: https://github.com/ElMoorish/AI-DJ-Software

Metin → şarkı arama: "dark energetic techno", "soft turkish ballad" gibi
serbest metin ile kütüphaneden eşleşen şarkıları bulur.

Gereksinim: pip install msclap  (Microsoft CLAP)
Ağır model (~900MB), ilk çalıştırmada indirilir.
"""

from pathlib import Path


def is_available() -> bool:
    try:
        import msclap
        return True
    except ImportError:
        return False


def search(query: str, song_paths: list[str], top_k: int = 5) -> list[dict]:
    """
    Metin sorgusuyla en benzer şarkıları döndürür.
    Returns: [{"path": ..., "score": ...}, ...]
    """
    if not is_available():
        raise ImportError(
            "CLAP kurulu değil. Kurmak için:\n"
            "  pip install msclap\n"
            "Not: ~900MB model indirilecek."
        )

    from msclap import CLAP

    model = CLAP(version="2023", use_cuda=False)

    # Audio embeddings — dosya başına ~1-3sn
    print(f"  {len(song_paths)} şarkı için embedding hesaplanıyor...")
    audio_embeddings = model.get_audio_embeddings(song_paths, resample=True)

    # Text embedding
    text_embeddings = model.get_text_embeddings([query])

    # Cosine similarity
    similarities = model.compute_similarity(audio_embeddings, text_embeddings)
    scores = similarities.squeeze().tolist()
    if isinstance(scores, float):
        scores = [scores]

    ranked = sorted(
        zip(song_paths, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return [{"path": p, "score": round(float(s), 4)} for p, s in ranked[:top_k]]
