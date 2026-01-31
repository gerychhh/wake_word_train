from __future__ import annotations
import os, sys
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]

def _p(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)

def check_exists(path: Path, kind: str = "path") -> bool:
    ok = path.exists()
    print(f"[{'OK' if ok else 'MISS'}] {kind:10s}: {_p(path)}")
    return ok

def main() -> int:
    print("=== wakeword doctor ===")
    print("root:", ROOT)
    print("python:", sys.version.replace("\n"," "))
    print("platform:", platform.platform())

    # Torch info (optional)
    try:
        import torch
        print("torch:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("cuda device:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("[WARN] torch not importable:", e)

    # Expected skeleton
    ok = True
    ok &= check_exists(ROOT / "openWakeWord" / "openwakeword" / "train.py", "train.py")
    ok &= check_exists(ROOT / "configs" / "beavis.yml", "config")
    check_exists(ROOT / "training_results", "dir")

    # Data expectations (not all mandatory)
    check_exists(ROOT / "data" / "my_data" / "beavis", "positive_dir")
    check_exists(ROOT / "data" / "backgrounds" / "musan", "musan_dir")
    check_exists(ROOT / "data" / "features" / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy", "ACAV100M")
    check_exists(ROOT / "data" / "features" / "validation_set_features.npy", "val_features")
    check_exists(ROOT / "data" / "piper", "piper_dir")


# openWakeWord internal models (downloaded by openwakeword.utils.download_models)
models_dir = ROOT / "openWakeWord" / "openwakeword" / "resources" / "models"
check_exists(models_dir / "melspectrogram.onnx", "melspec")
check_exists(models_dir / "embedding_model.onnx", "embed")
check_exists(models_dir / "silero_vad.onnx", "vad")


    print("=== done ===")
    return 0 if ok else 2

if __name__ == "__main__":
    raise SystemExit(main())
