"""
DJtransGAN entegrasyonu — GAN tabanlı AI geçiş üretimi.
Kaynak: https://github.com/ChenPaulYu/DJtransGAN (MIT License)

Pre-trained model otomatik indirilir (ilk çalıştırmada).
PyTorch gerektirmez — subprocess ile inference.py çağrılır.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

DJTRANSGAN_DIR = Path(__file__).parent.parent / "engines" / "DJtransGAN"
INFERENCE_SCRIPT = DJTRANSGAN_DIR / "script" / "inference.py"
MODEL_PATH = DJTRANSGAN_DIR / "djtransgan_minmax.pt"


def is_available() -> bool:
    """DJtransGAN kurulu ve model mevcut mu?"""
    return INFERENCE_SCRIPT.exists()


def install() -> bool:
    """DJtransGAN reposunu engines/ altına klonlar."""
    engines_dir = DJTRANSGAN_DIR.parent
    engines_dir.mkdir(parents=True, exist_ok=True)

    if DJTRANSGAN_DIR.exists():
        print("DJtransGAN zaten kurulu.")
        return True

    print("DJtransGAN klonlanıyor...")
    result = subprocess.run(
        ["git", "clone", "--depth=1",
         "https://github.com/ChenPaulYu/DJtransGAN.git",
         str(DJTRANSGAN_DIR)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Clone hatası: {result.stderr}")
        return False

    # Bağımlılıkları kur (sadece inference için gerekli olanlar)
    print("Bağımlılıklar kuruluyor (torch, torchaudio)...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "torch", "torchaudio", "resampy", "madmom"],
        check=False
    )
    print("DJtransGAN kurulumu tamamlandı.")
    return True


def generate_transition(
    file_a: str,
    file_b: str,
    cue_a: float,
    cue_b: float,
    output_dir: str = "output",
) -> str | None:
    """
    DJtransGAN inference.py'yi subprocess ile çağırır.
    Başarılıysa çıktı dosyasının yolunu döndürür, değilse None.
    """
    if not is_available():
        print("DJtransGAN kurulu değil. Önce: python main.py install-engine")
        return None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(INFERENCE_SCRIPT),
        "--prev_track", str(file_a),
        "--next_track", str(file_b),
        "--prev_cue", str(int(cue_a)),
        "--next_cue", str(int(cue_b)),
        "--out_dir", str(out_dir),
        "--download",  # pre-trained model yoksa indir
    ]

    print(f"DJtransGAN çalıştırılıyor...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(DJTRANSGAN_DIR))

    if result.returncode != 0:
        print(f"DJtransGAN hatası:\n{result.stderr[-500:]}")
        return None

    # inference.py çıktı dosyasını bul
    candidates = sorted(out_dir.glob("*.wav"), key=os.path.getmtime, reverse=True)
    if candidates:
        return str(candidates[0])

    print("DJtransGAN çıktı dosyası bulunamadı.")
    return None
