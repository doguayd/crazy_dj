"""
crazy_dj CLI

Komutlar:
  analyze <file>                      — Tek şarkı analiz et
  compare <file_a> <file_b>           — İki şarkıyı karşılaştır
  mix <file_a> <file_b>               — Transition klip üret
    --fade 8                            Crossfade süresi (saniye)
    --effect sweep|echo|deep|energetic|clean
    --engine auto|djtransgan|builtin    Geçiş motoru
    --out output/mix.flac
  scan <directory>                    — Tüm klasörü analiz et
  match                               — DB'den uyumlu çiftleri bul
    --top 10                            Kaç çift gösterilsin
    --min-score 0.55                    Minimum uyum skoru
  setlist                             — Otomatik setlist oluştur
    --length 8                          Kaç şarkı
    --start "Şarkı adı"                 Başlangıç şarkısı (opsiyonel)
  automix                             — Setlist oluşturup tüm geçişleri üret
    --length 8
    --effect sweep
    --engine auto|djtransgan|builtin
    --fade 8
  search <query>                      — Metin ile şarkı ara (CLAP gerekli)
    --top 5
  install-engine                      — DJtransGAN'ı indir ve kur
  engines                             — Kurulu engine'leri göster
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Windows terminali UTF-8 yapılandır
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from analyzer.audio_analyzer import analyze
from analyzer.similarity import compare as compare_songs
from analyzer.matcher import find_compatible_pairs, build_setlist, compatibility_score
from mixer.crossfader import mix_transition
from db import store


# ── Helpers ───────────────────────────────────────────────────────────────────

def _song_name(path: str) -> str:
    return Path(path).stem


def _ensure_analyzed(file_path: str):
    resolved = str(Path(file_path).resolve())
    cached = store.load(resolved)
    if cached:
        return cached
    print(f"  Analyzing {Path(file_path).name} ...", end=" ", flush=True)
    result = analyze(file_path)
    store.save(result)
    print(f"{result.bpm:.1f} BPM, {result.key}")
    return result


def _do_mix(file_a, file_b, cue_a, cue_b, fade_sec, effect, engine, out_path):
    """Engine seçimine göre geçiş üretir."""
    if engine in ("djtransgan", "auto"):
        from mixer.djtransgan_engine import is_available, generate_transition
        if is_available():
            print(f"  Engine: DJtransGAN (AI)")
            result = generate_transition(file_a, file_b, cue_a, cue_b,
                                         output_dir=str(Path(out_path).parent))
            if result:
                return result
            print("  DJtransGAN başarısız, builtin'e geçiliyor...")
        elif engine == "djtransgan":
            print("DJtransGAN kurulu değil. Kur: python main.py install-engine")
            sys.exit(1)

    print(f"  Engine: builtin  |  Efekt: {effect}")
    return mix_transition(file_a, cue_a, file_b, cue_b,
                          fade_sec=fade_sec, output_path=out_path, effect=effect)


# ── Komutlar ──────────────────────────────────────────────────────────────────

def cmd_analyze(args):
    store.init_db()
    cached = store.load(str(Path(args.file).resolve()))
    if cached and not args.force:
        print(json.dumps({"cached": True, "bpm": cached.bpm, "key": cached.key,
                          "energy": cached.energy, "duration": cached.duration}, indent=2))
        return
    print(f"Analyzing {args.file} ...")
    result = analyze(args.file)
    store.save(result)
    print(json.dumps({"bpm": result.bpm, "key": result.key,
                      "energy": result.energy, "duration": result.duration}, indent=2))


def cmd_compare(args):
    store.init_db()
    a = _ensure_analyzed(args.file_a)
    b = _ensure_analyzed(args.file_b)
    result = compare_songs(a, b)
    score = compatibility_score(a, b)
    print(f"\n{'─'*55}")
    print(f"  {_song_name(a.path)[:24]}  ↔  {_song_name(b.path)[:24]}")
    print(f"{'─'*55}")
    print(f"  Uyum skoru   : {score:.2f} / 1.00")
    print(f"  Key uyumu    : {'✓ Evet' if result['key_compatible'] else '✗ Hayır'}  ({a.key} ↔ {b.key})")
    print(f"  BPM farkı    : {result['bpm_distance']:.1f}  ({a.bpm} ↔ {b.bpm})")
    print(f"  Chroma benzer: {result['chroma_similarity']:.2f}")
    print(f"\n  Sync noktaları:")
    for sp in result["sync_points"]:
        print(f"    A:{sp['time_a']:.1f}s → B:{sp['time_b']:.1f}s  (skor {sp['score']:.2f})")


def cmd_mix(args):
    store.init_db()
    a = _ensure_analyzed(args.file_a)
    b = _ensure_analyzed(args.file_b)

    comparison = compare_songs(a, b)
    sync = comparison["sync_points"]
    if not sync:
        print("Sync noktası bulunamadı.", file=sys.stderr)
        sys.exit(1)

    best = sync[0]
    print(f"\nEn iyi sync: A @ {best['time_a']}s → B @ {best['time_b']}s")

    out = _do_mix(args.file_a, args.file_b, best["time_a"], best["time_b"],
                  args.fade, args.effect, args.engine, args.out)
    print(f"Kaydedildi: {out}")


def cmd_scan(args):
    store.init_db()
    directory = Path(args.directory)
    files = (list(directory.rglob("*.flac")) + list(directory.rglob("*.mp3"))
           + list(directory.rglob("*.wav")) + list(directory.rglob("*.ogg")))
    print(f"{len(files)} dosya bulundu: {directory}\n")
    for f in files:
        cached = store.load(str(f.resolve()))
        if cached:
            print(f"  [cache] {f.name:<55} {cached.bpm:.1f} BPM  {cached.key}")
            continue
        print(f"  Analyzing {f.name} ...", end=" ", flush=True)
        try:
            result = analyze(str(f))
            store.save(result)
            print(f"{result.bpm:.1f} BPM, {result.key}")
        except Exception as e:
            print(f"HATA: {e}")


def cmd_match(args):
    store.init_db()
    songs = store.load_all()
    if not songs:
        print("DB boş — önce 'scan' çalıştır.")
        return

    print(f"\n{len(songs)} şarkı karşılaştırılıyor...\n")
    pairs = find_compatible_pairs(songs, min_score=args.min_score, top_n=args.top)

    if not pairs:
        print(f"Min skor {args.min_score} üzerinde uyumlu çift bulunamadı.")
        return

    print(f"  {'#':<3}  {'SKOR':<6}  {'KEY':<4}  {'BPM FARKI':<10}  ŞARKILAR")
    print(f"  {'─'*72}")
    for i, p in enumerate(pairs, 1):
        key_icon = "✓" if p["key_compatible"] else "~"
        bpm_diff = abs(p["bpm_a"] - p["bpm_b"])
        name_a = _song_name(p["song_a"].path)[:26]
        name_b = _song_name(p["song_b"].path)[:26]
        print(f"  {i:<3}  {p['score']:.2f}   {key_icon}    {bpm_diff:<10.1f}  {name_a}  ↔  {name_b}")


def cmd_setlist(args):
    store.init_db()
    songs = store.load_all()
    if not songs:
        print("DB boş — önce 'scan' çalıştır.")
        return

    start = None
    if args.start:
        query = args.start.lower()
        matches = [s for s in songs if query in _song_name(s.path).lower()]
        if matches:
            start = matches[0]
            print(f"Başlangıç: {_song_name(start.path)}")
        else:
            print(f"'{args.start}' bulunamadı, en enerjik şarkıdan başlanıyor.")

    setlist = build_setlist(songs, start_song=start, length=args.length)

    print(f"\n{'─'*58}")
    print(f"  SETLIST ({len(setlist)} şarkı)")
    print(f"{'─'*58}")
    for i, song in enumerate(setlist, 1):
        print(f"  {i:>2}. {_song_name(song.path):<45} {song.bpm:.1f} BPM  {song.key}")
        if i < len(setlist):
            score = compatibility_score(song, setlist[i])
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            print(f"       ↓ [{bar}] {score:.2f}")
    print(f"{'─'*58}")


def cmd_automix(args):
    store.init_db()
    songs = store.load_all()
    if not songs:
        print("DB boş — önce 'scan' çalıştır.")
        return

    setlist = build_setlist(songs, length=args.length)
    print(f"\nSetlist ({len(setlist)} şarkı) hazır, mix üretiliyor...\n")

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    for i in range(len(setlist) - 1):
        a = setlist[i]
        b = setlist[i + 1]
        name = (f"{i+1:02d}_{_song_name(a.path)[:18]}_to_{_song_name(b.path)[:18]}.flac"
                .replace(" ", "_"))
        out_path = str(out_dir / name)

        comparison = compare_songs(a, b)
        sync = comparison["sync_points"]
        if not sync:
            print(f"  [{i+1}] Sync yok, atlanıyor.")
            continue

        best = sync[0]
        score = compatibility_score(a, b)
        print(f"  [{i+1}] {_song_name(a.path)[:22]} → {_song_name(b.path)[:22]}  (uyum: {score:.2f})")

        try:
            result = _do_mix(a.path, b.path, best["time_a"], best["time_b"],
                             args.fade, args.effect, args.engine, out_path)
            print(f"       → {result}")
        except Exception as e:
            print(f"       HATA: {e}")

    print(f"\nTamamlandı! → {out_dir.resolve()}")


def cmd_search(args):
    store.init_db()
    from analyzer.clap_search import is_available, search as clap_search

    if not is_available():
        print("CLAP kurulu değil. Kurmak için:\n  pip install msclap")
        print("Not: ~900MB model ilk çalıştırmada indirilir.")
        sys.exit(1)

    songs = store.load_all()
    if not songs:
        print("DB boş — önce 'scan' çalıştır.")
        return

    paths = [s.path for s in songs]
    print(f"'{args.query}' için {len(paths)} şarkı arasında aranıyor...\n")
    results = clap_search(args.query, paths, top_k=args.top)

    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['score']:.3f}]  {_song_name(r['path'])}")


def cmd_install_engine(args):
    from mixer.djtransgan_engine import install
    install()


def cmd_engines(args):
    from mixer.djtransgan_engine import is_available as djtransgan_ok
    from analyzer.clap_search import is_available as clap_ok

    print("\n  Engine / Özellik       Durum")
    print("  " + "─" * 35)
    print(f"  DJtransGAN (AI mix)    {'✓ Kurulu' if djtransgan_ok() else '✗ Kurulu değil  →  python main.py install-engine'}")
    print(f"  CLAP (semantik arama)  {'✓ Kurulu' if clap_ok() else '✗ Kurulu değil  →  pip install msclap'}")

    try:
        import madmom
        madmom_ok = True
    except ImportError:
        madmom_ok = False
    print(f"  madmom (downbeat)      {'✓ Kurulu' if madmom_ok else '✗ Kurulu değil  →  pip install madmom'}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(prog="crazy_dj", description="DJ simulation & auto-mixer")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze");  p.add_argument("file"); p.add_argument("--force", action="store_true")

    p = sub.add_parser("compare");  p.add_argument("file_a"); p.add_argument("file_b")

    p = sub.add_parser("mix")
    p.add_argument("file_a"); p.add_argument("file_b")
    p.add_argument("--fade", type=float, default=8.0)
    p.add_argument("--effect", default="sweep", choices=["clean","sweep","echo","deep","energetic"])
    p.add_argument("--engine", default="auto", choices=["auto","djtransgan","builtin"])
    p.add_argument("--out", default="output/transition.flac")

    p = sub.add_parser("scan");     p.add_argument("directory")

    p = sub.add_parser("match")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--min-score", type=float, default=0.55)

    p = sub.add_parser("setlist")
    p.add_argument("--length", type=int, default=8)
    p.add_argument("--start", type=str, default=None)

    p = sub.add_parser("automix")
    p.add_argument("--length", type=int, default=8)
    p.add_argument("--effect", default="sweep", choices=["clean","sweep","echo","deep","energetic"])
    p.add_argument("--engine", default="auto", choices=["auto","djtransgan","builtin"])
    p.add_argument("--fade", type=float, default=8.0)

    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--top", type=int, default=5)

    sub.add_parser("install-engine")
    sub.add_parser("engines")

    args = parser.parse_args()
    {
        "analyze":        cmd_analyze,
        "compare":        cmd_compare,
        "mix":            cmd_mix,
        "scan":           cmd_scan,
        "match":          cmd_match,
        "setlist":        cmd_setlist,
        "automix":        cmd_automix,
        "search":         cmd_search,
        "install-engine": cmd_install_engine,
        "engines":        cmd_engines,
    }[args.command](args)


if __name__ == "__main__":
    main()
