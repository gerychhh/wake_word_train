from __future__ import annotations
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "data" / "_generated_clips"

def main():
    if GEN.exists():
        shutil.rmtree(GEN)
        GEN.mkdir(parents=True, exist_ok=True)
        (GEN / ".gitkeep").write_text("Temporary generated WAV clips (auto-clean).\n", encoding="utf-8")
        print("Cleaned:", GEN)
    else:
        print("Nothing to clean:", GEN)

if __name__ == "__main__":
    main()
