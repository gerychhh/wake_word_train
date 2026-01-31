from __future__ import annotations

import os
import argparse
from pathlib import Path
import shutil
import numpy as np

from scipy.io import wavfile
from scipy.signal import resample_poly


def to_mono(x: np.ndarray) -> np.ndarray:
    """Convert audio to mono float32 [-1..1]."""
    if x.ndim == 1:
        return x.astype(np.float32)

    x = x.astype(np.float32)
    return x.mean(axis=1)


def int16_from_float(x: np.ndarray) -> np.ndarray:
    """Clamp float [-1..1] to int16."""
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


def normalize_to_float(x: np.ndarray) -> np.ndarray:
    """Convert wav read data to float32 in [-1..1] if needed."""
    if x.dtype == np.int16:
        return x.astype(np.float32) / 32768.0

    if x.dtype == np.int32:
        maxv = np.max(np.abs(x))
        if maxv == 0:
            return x.astype(np.float32)
        return (x.astype(np.float32) / maxv).astype(np.float32)

    if x.dtype == np.uint8:
        return ((x.astype(np.float32) - 128.0) / 128.0).astype(np.float32)

    return x.astype(np.float32)


def resample_audio(x_float: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample float32 mono audio from sr_in to sr_out using polyphase."""
    if sr_in == sr_out:
        return x_float

    from math import gcd
    g = gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g

    y = resample_poly(x_float, up=up, down=down).astype(np.float32)
    return y


def process_wav(path: Path, target_sr: int, backup: bool, dry_run: bool) -> tuple[bool, str]:
    """
    Returns (changed, message)
    """
    try:
        sr, data = wavfile.read(str(path))
    except Exception as e:
        return (False, f"[SKIP] read error: {e}")

    if not isinstance(sr, (int, np.integer)):
        return (False, f"[SKIP] invalid sample rate: {sr}")

    already_mono = (data.ndim == 1) or (data.ndim == 2 and data.shape[1] == 1)
    already_int16 = (data.dtype == np.int16)
    already_sr = (int(sr) == int(target_sr))

    if already_sr and already_mono and already_int16:
        return (False, "[OK] already 16k mono int16")

    x = normalize_to_float(data)
    x = to_mono(x)
    x = resample_audio(x, int(sr), int(target_sr))

    if x.size < 10:
        return (False, "[SKIP] too short after processing")

    out = int16_from_float(x)

    if dry_run:
        return (True, f"[DRY] would convert sr {sr}->{target_sr}")

    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            try:
                shutil.copy2(str(path), str(bak))
            except Exception as e:
                return (False, f"[SKIP] backup failed: {e}")

    try:
        wavfile.write(str(path), target_sr, out)
    except Exception as e:
        return (False, f"[SKIP] write error: {e}")

    return (True, f"[FIX] sr {sr}->{target_sr}")


def fix_folder(
    root: str | Path,
    target_sr: int = 16000,
    backup: bool = False,
    dry_run: bool = False,
    delete_bak: bool = True,
    only_substring: str = "",
    quiet: bool = True,
) -> dict:
    """
    Fix all wav files inside folder (recursive).
    Returns stats dict so you can log it in train.py.
    """
    root = Path(root).resolve()
    only = only_substring.strip().lower()

    wavs: list[Path] = []
    for p in root.rglob("*.wav"):
        s = str(p).lower()
        if only and only not in s:
            continue
        wavs.append(p)

    stats = {
        "root": str(root),
        "total": len(wavs),
        "ok": 0,
        "fixed": 0,
        "skipped": 0,
        "deleted_bak": 0,
        "dry_run": dry_run,
        "backup": backup,
        "target_sr": target_sr,
    }

    # ✅ просто счётчик 1,2,3...
    counter = 0

    for wav_path in wavs:
        counter += 1
        changed, msg = process_wav(wav_path, target_sr=target_sr, backup=backup, dry_run=dry_run)

        if msg.startswith("[OK]"):
            stats["ok"] += 1
        elif msg.startswith("[SKIP]"):
            stats["skipped"] += 1
        else:
            stats["fixed"] += 1

        # Удаляем .bak если нужно
        if delete_bak:
            bak = wav_path.with_suffix(wav_path.suffix + ".bak")
            if bak.exists():
                try:
                    bak.unlink()
                    stats["deleted_bak"] += 1
                except:
                    pass

        if not quiet:
            # если хочешь видеть подробности — можно выключить quiet
            print(counter, wav_path.name, msg)
        else:
            # по твоему запросу: просто число
            print(counter)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert WAV files to 16kHz mono int16 (fix openWakeWord sample rate errors)."
    )
    parser.add_argument("--root", type=str, default=".", help="Root folder to scan (default: current dir)")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate (default: 16000)")
    parser.add_argument("--no-backup", action="store_true", help="Disable .bak backup creation")
    parser.add_argument("--dry", action="store_true", help="Dry run (do not modify files)")
    parser.add_argument("--only", type=str, default="", help="Only process paths containing this substring")
    parser.add_argument("--keep-bak", action="store_true", help="Do NOT delete .bak files")
    parser.add_argument("--verbose", action="store_true", help="Print filename + message (not only counter)")
    args = parser.parse_args()

    stats = fix_folder(
        root=args.root,
        target_sr=args.sr,
        backup=not args.no_backup,
        dry_run=args.dry,
        delete_bak=not args.keep_bak,
        only_substring=args.only,
        quiet=not args.verbose,
    )

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"  ROOT      : {stats['root']}")
    print(f"  TOTAL     : {stats['total']}")
    print(f"  OK        : {stats['ok']}")
    print(f"  FIXED     : {stats['fixed']}")
    print(f"  SKIPPED   : {stats['skipped']}")
    print(f"  DEL .BAK  : {stats['deleted_bak']}")
    print(f"  TARGET SR : {stats['target_sr']}")
    print(f"  DRY RUN   : {stats['dry_run']}")
    print(f"  BACKUP    : {stats['backup']}")
    print("=" * 80)

    if args.dry:
        print("\nDry run завершён. Чтобы реально исправить файлы — запусти без --dry.")


if __name__ == "__main__":
    main()
