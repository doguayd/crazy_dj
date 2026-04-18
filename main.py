"""
crazy_dj CLI — analyze songs, compare them, and generate transition clips.

Usage:
  python main.py analyze <file>
  python main.py compare <file_a> <file_b>
  python main.py mix <file_a> <file_b> [--fade 8] [--out output/mix.flac]
  python main.py scan <directory>
"""

import argparse
import json
import sys
from pathlib import Path

from analyzer.audio_analyzer import analyze
from analyzer.similarity import compare
from mixer.crossfader import mix_transition
from db import store


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
    a = store.load(str(Path(args.file_a).resolve())) or analyze(args.file_a)
    b = store.load(str(Path(args.file_b).resolve())) or analyze(args.file_b)
    store.save(a)
    store.save(b)
    result = compare(a, b)
    print(json.dumps(result, indent=2))


def cmd_mix(args):
    store.init_db()
    a = store.load(str(Path(args.file_a).resolve())) or analyze(args.file_a)
    b = store.load(str(Path(args.file_b).resolve())) or analyze(args.file_b)
    store.save(a)
    store.save(b)

    comparison = compare(a, b)
    sync = comparison["sync_points"]
    if not sync:
        print("No sync points found.", file=sys.stderr)
        sys.exit(1)

    best = sync[0]
    print(f"Best sync: track A @ {best['time_a']}s → track B @ {best['time_b']}s "
          f"(score {best['score']})")

    out = mix_transition(
        args.file_a, best["time_a"],
        args.file_b, best["time_b"],
        fade_sec=args.fade,
        output_path=args.out,
    )
    print(f"Transition saved: {out}")


def cmd_scan(args):
    store.init_db()
    directory = Path(args.directory)
    files = list(directory.rglob("*.flac")) + list(directory.rglob("*.mp3")) \
          + list(directory.rglob("*.wav")) + list(directory.rglob("*.ogg"))
    print(f"Found {len(files)} files in {directory}")
    for f in files:
        cached = store.load(str(f.resolve()))
        if cached:
            print(f"  [cached] {f.name}")
            continue
        print(f"  Analyzing {f.name} ...", end=" ", flush=True)
        try:
            result = analyze(str(f))
            store.save(result)
            print(f"{result.bpm:.1f} BPM, {result.key}")
        except Exception as e:
            print(f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(prog="crazy_dj")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze")
    p_analyze.add_argument("file")
    p_analyze.add_argument("--force", action="store_true")

    p_compare = sub.add_parser("compare")
    p_compare.add_argument("file_a")
    p_compare.add_argument("file_b")

    p_mix = sub.add_parser("mix")
    p_mix.add_argument("file_a")
    p_mix.add_argument("file_b")
    p_mix.add_argument("--fade", type=float, default=8.0)
    p_mix.add_argument("--out", default="output/transition.flac")

    p_scan = sub.add_parser("scan")
    p_scan.add_argument("directory")

    args = parser.parse_args()
    {
        "analyze": cmd_analyze,
        "compare": cmd_compare,
        "mix": cmd_mix,
        "scan": cmd_scan,
    }[args.command](args)


if __name__ == "__main__":
    main()
