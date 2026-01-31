from __future__ import annotations

import os
import random
import subprocess
import logging
import platform
from pathlib import Path
from typing import List, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# -----------------------------
# Helpers
# -----------------------------
def _resolve_piper_exe(piper_executable: Optional[str]) -> str:
    """
    1) Если передали явно -> используем
    2) Иначе берем из env PIPER_EXE
    3) Иначе просто "piper.exe" (если в PATH)
    """
    if piper_executable:
        return str(piper_executable)

    env_exe = os.environ.get("PIPER_EXE")
    if env_exe:
        return str(env_exe)

    return "piper.exe" if platform.system() == "Windows" else "piper"


def _resolve_models(model: Union[str, List[str], Tuple[str, ...]]) -> List[Tuple[str, str]]:
    """
    Поддержка:
    - model = путь к .onnx (рядом должен быть .onnx.json)
    - model = папка (внутри много .onnx + .onnx.json) -> будет случайный выбор модели каждый wav
    Возвращает список пар: (onnx_path, onnx_json_path)
    """
    # Если пришел список/кортеж моделей/папок — разрешаем каждую и объединяем
    if isinstance(model, (list, tuple)):
        merged: List[Tuple[str, str]] = []
        seen = set()
        for m in model:
            for onnx, cfg in _resolve_models(m):
                key = (onnx, cfg)
                if key not in seen:
                    merged.append((onnx, cfg))
                    seen.add(key)
        if not merged:
            raise FileNotFoundError(f"Не найдено ни одной валидной пары .onnx + .onnx.json в списке: {model}")
        return merged

    model_path = Path(model)

    if model_path.is_dir():
        onnx_files = sorted(model_path.glob("*.onnx"))
        pairs: List[Tuple[str, str]] = []
        for onnx in onnx_files:
            cfg = Path(str(onnx) + ".json")  # *.onnx.json
            if cfg.exists():
                pairs.append((str(onnx), str(cfg)))

        if not pairs:
            raise FileNotFoundError(
                f"В папке нет пар *.onnx + *.onnx.json: {model}\n"
                f"Ожидается например:\n"
                f"  ru_RU-dmitri-medium.onnx\n"
                f"  ru_RU-dmitri-medium.onnx.json"
            )
        return pairs

    if model_path.is_file():
        if model_path.suffix.lower() != ".onnx":
            raise FileNotFoundError(
                f"Ожидается .onnx файл или папка с ними, а пришло: {model}"
            )

        cfg = Path(str(model_path) + ".json")  # *.onnx.json
        if not cfg.exists():
            raise FileNotFoundError(
                f"Не найден конфиг: {cfg}\n"
                f"Рядом с моделью должен быть файл *.onnx.json"
            )

        return [(str(model_path), str(cfg))]

    raise FileNotFoundError(f"piper model not found: {model}")


def _safe_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _run_one(
    piper_exe: str,
    models: List[Tuple[str, str]],
    texts: List[str],
    output_dir: str,
    out_name: str,
    noise_scales: List[float],
    noise_scale_ws: List[float],
    length_scales: List[float],
    use_cuda: bool,
) -> str:
    """
    Генерация одного wav.
    Возвращает имя голоса (stem) для отображения в прогрессе.
    """
    t = random.choice(texts)

    length_scale = random.choice(length_scales)
    noise_scale = random.choice(noise_scales)
    noise_w = random.choice(noise_scale_ws)

    onnx_path, json_path = random.choice(models)

    out_path = os.path.join(output_dir, out_name)

    cmd = [
        piper_exe,
        "-m", onnx_path,
        "-c", json_path,
        "-f", out_path,
        "--length-scale", str(length_scale),
        "--noise-scale", str(noise_scale),
        "--noise-w-scale", str(noise_w),
    ]

    # CUDA флаг (если твой piper.exe его поддерживает)
    if use_cuda:
        cmd.append("--cuda")

    import tempfile

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
        f.write(t + "\n")
        in_path = f.name

    try:
        subprocess.run(
            cmd + ["-i", in_path],
            check=True
        )
    finally:
        try:
            os.remove(in_path)
        except:
            pass

    return Path(onnx_path).name


# -----------------------------
# Main API (openWakeWord compatible)
# -----------------------------
def generate_samples(
    model: str,
    text,
    max_samples: int,
    batch_size: int = 1,
    noise_scales=None,
    noise_scale_ws=None,
    length_scales=None,
    output_dir: str = ".",
    auto_reduce_batch_size: bool = True,
    file_names=None,
    piper_executable: str | None = None,
):
    """
    ✅ Piper CLI генерация WAV + ✅ МНОГОПОТОК + ✅ tqdm прогресс

    Совместимо с openWakeWord train.py

    model = путь к .onnx ИЛИ папка с множеством .onnx (+.onnx.json)
    text  = строка или список строк
    """

    # workers задаётся так:
    # 1) из env PIPER_WORKERS
    # 2) иначе fallback на batch_size
    # 3) иначе 4
    workers = _safe_int_env("PIPER_WORKERS", default=batch_size if batch_size and batch_size > 0 else 4)
    workers = max(1, workers)

    use_cuda = os.environ.get("PIPER_CUDA", "0").strip() == "1"

    piper_exe = _resolve_piper_exe(piper_executable)
    models = _resolve_models(model)

    if isinstance(text, (list, tuple)):
        texts_raw = list(text)
    else:
        texts_raw = [text]

    # чистим None/пустые/пробелы
    texts = []
    for x in texts_raw:
        if x is None:
            continue
        s = str(x).strip()
        if not s:
            continue
        texts.append(s)

    if not texts:
        raise ValueError("[PIPER] ERROR: text list is empty after cleaning (all were None/empty)")

    noise_scales = noise_scales or [0.667]
    noise_scale_ws = noise_scale_ws or [0.8]
    length_scales = length_scales or [1.0]

    os.makedirs(output_dir, exist_ok=True)

    total = int(max_samples)
    if total <= 0:
        return None

    # имена файлов
    if file_names and len(file_names) >= total:
        names = list(file_names[:total])
    else:
        names = [f"{i:06d}.wav" for i in range(total)]

    folder_name = Path(output_dir).name
    voice_info = f"{len(models)} voice(s)"
    mode = "CUDA" if use_cuda else "CPU"
    desc = f"PIPER → {folder_name} ({voice_info}, {mode}, workers={workers})"

    # --- Многопоток на Windows лучше ThreadPoolExecutor ---
    # потому что subprocess.run сам запускает отдельный процесс,
    # GIL не мешает, а ProcessPool даёт больше гемора.
    futures = []
    ok = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for out_name in names:
            futures.append(
                ex.submit(
                    _run_one,
                    piper_exe,
                    models,
                    texts,
                    output_dir,
                    out_name,
                    noise_scales,
                    noise_scale_ws,
                    length_scales,
                    use_cuda,
                )
            )

        iterator = as_completed(futures)

        if tqdm is not None:
            iterator = tqdm(iterator, total=total, desc=desc, unit="wav", dynamic_ncols=True)

        for fut in iterator:
            try:
                voice_name = fut.result()
                ok += 1
                if tqdm is not None and hasattr(iterator, "set_postfix_str"):
                    iterator.set_postfix_str(voice_name[:28])
            except Exception as e:
                logging.error(f"[PIPER] generation failed: {e}")
                continue

    logging.info(f"[PIPER] DONE: {ok}/{total} wav generated in {output_dir}")
    return None
