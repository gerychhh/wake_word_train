# beavis-wakeword-trainer
**Portable wake-word training workspace (openWakeWord) — Linux & Windows**  
**EN + RU (bilingual) / Двуязычный README**

---

## TL;DR (EN)
1) Create venv → 2) Install PyTorch → 3) Install deps → 4) Put data into `data/` → 5) Download internal OWw models → 6) Train.

## TL;DR (RU)
1) Создай venv → 2) Поставь PyTorch → 3) Поставь зависимости → 4) Разложи данные в `data/` → 5) Скачай внутренние модели OWw → 6) Запускай обучение.

---

# English

## 1. What this repo is (and what it is not)

This repository is a **clean and portable** training workspace for **openWakeWord** custom wake-word models.

### It includes
- A trimmed vendored copy of `openWakeWord/` (source + training script).
- A minimal Piper-based clip generator in `tools/piper_sample_generator/`.
- Portable config(s) under `configs/` that use **relative paths** where possible.
- A clear convention: **code/configs tracked**, **data + results not tracked**.

### It does NOT include
- Large datasets (MUSAN, ACAV100M `.npy`, your audio recordings).
- Piper voice models (`*.onnx` + `*.onnx.json`).
- Training outputs (`*.onnx`, generated clips, feature dumps).

Those must be placed under `data/` and `training_results/` locally.

---

## 2. Folder layout (what goes where, and why)

### High-level tree
```
repo/
├── configs/                         # training configs (portable)
│   ├── beavis.yml
│   └── custom_model_windows_original.yml
│
├── openWakeWord/                    # vendored openWakeWord package
│   └── openwakeword/                # python package
│       ├── train.py                 # training entrypoint
│       ├── data.py                  # augmentation + adversarial text
│       ├── utils.py                 # feature extractor helper, downloads
│       └── resources/
│           └── models/              # internal OWw ONNX models (downloaded)
│
├── tools/
│   └── piper_sample_generator/      # Piper CLI wrapper to generate clips
│       └── generate_samples.py
│
├── scripts/
│   └── doctor.py                    # environment checks
│
├── requirements/                    # pinned dependency sets
│   ├── py310.txt
│   └── py310-tflite.txt             # only if you need ONNX→TFLite
│
├── data/                            # NOT tracked: datasets + your wavs + voices
│   ├── README_DATA.md
│   ├── my_data/
│   │   └── beavis/                  # your positive wavs
│   ├── backgrounds/
│   │   └── musan/                   # noise/speech/music
│   ├── features/
│   │   ├── openwakeword_features_ACAV100M_2000_hrs_16bit.npy
│   │   └── validation_set_features.npy
│   └── piper/
│       └── voices/                  # Piper voice models folder (multi-voice)
│           ├── ru_RU-dmitri-medium.onnx
│           ├── ru_RU-dmitri-medium.onnx.json
│           ├── ru_RU-denis-medium.onnx
│           ├── ru_RU-denis-medium.onnx.json
│           └── ...
│
└── training_results/                # NOT tracked: generated clips, features, models
    └── beavis/
        ├── positive_train/          # generated + fixed + augmented
        ├── positive_test/
        ├── negative_train/
        ├── negative_test/
        ├── *_features_train.npy
        ├── *_features_test.npy
        ├── beavis.onnx              # final model
        └── logs/                    # logs, stats, checkpoints
```

### What belongs where (detailed)

#### `configs/`
- **Purpose:** source-of-truth training configuration(s).
- **What goes here:** YAML configs only.
- **Typical problems:**
  - Absolute paths break portability → prefer relative paths and environment variables.
  - Mis-typed paths cause missing file errors.

#### `openWakeWord/openwakeword/resources/models/`
- **Purpose:** internal ONNX models used by openWakeWord for feature extraction (e.g., `melspectrogram.onnx`, embedding models, VAD model).
- **How it gets populated:** one-time download via:
  ```bash
  python -c "import openwakeword.utils as u; u.download_models(['_features_only_'])"
  ```
- **Typical problems:**
  - `NO_SUCHFILE ... melspectrogram.onnx` → you didn’t download these yet.

#### `tools/piper_sample_generator/`
- **Purpose:** generate synthetic positive samples using Piper (TTS) to increase data volume.
- **What goes here:** code only.
- **Typical problems:**
  - Piper executable missing/not in PATH.
  - Piper model `.onnx.json` missing.
  - Missing Python deps used by Piper package (example: `pathvalidate`).

#### `data/` (NOT in git)
- **Purpose:** all heavy assets, datasets, and local-only resources.
- **Do not commit this folder.**
- **Must contain:** your positives, MUSAN backgrounds, ACAV `.npy`, (optional) Piper voices.

#### `training_results/` (NOT in git)
- **Purpose:** everything produced by the pipeline:
  generated clips, augmented data, computed `.npy` features, final models, logs.
- **Typical problems:**
  - Missing directories if the script expects them to exist.
  - Partial runs leave stale data → use `--overwrite` when you want a clean rebuild.

---

## 3. Data requirements (mandatory vs optional)

### Mandatory for training
1) **Your positive wavs**
```
data/my_data/beavis/*.wav
```

2) **MUSAN backgrounds**
```
data/backgrounds/musan/noise/
data/backgrounds/musan/speech/
data/backgrounds/musan/music/
```

3) **ACAV100M feature file (big `.npy`)**
```
data/features/openwakeword_features_ACAV100M_2000_hrs_16bit.npy
```

### Recommended (for false-positive validation)
4) Validation features file
```
data/features/validation_set_features.npy
```

### Optional (only if you use `--generate_clips`)
5) Piper voice models (single voice OR folder with multiple voices)
```
data/piper/voices/*.onnx
data/piper/voices/*.onnx.json
```

---

## 4. Setup (Python 3.10)

### 4.1 Create venv

**Linux**
```bash
python3.10 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip wheel setuptools
```

> If `python` command is missing on Ubuntu, use `python3` everywhere or install:
> `sudo apt install -y python-is-python3`

**Windows (PowerShell)**
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip wheel setuptools
```

### 4.2 Install PyTorch first (CPU or CUDA)
Use the official selector:
- https://pytorch.org/get-started/locally/

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 4.3 Install dependencies
```bash
pip install -r requirements/py310.txt
pip install -e ./openWakeWord
```

### 4.4 Download internal openWakeWord models (one time)
```bash
python -c "import openwakeword.utils as u; u.download_models(['_features_only_'])"
```

---

## 5. Running the pipeline

### 5.1 Most stable: train ONNX model
```bash
python openWakeWord/openwakeword/train.py   --training_config configs/beavis.yml   --train_model   --log_level INFO
```

### 5.2 Generate synthetic positives (Piper)
```bash
python openWakeWord/openwakeword/train.py   --training_config configs/beavis.yml   --generate_clips   --log_level INFO
```

### 5.3 Augment + compute features
```bash
python openWakeWord/openwakeword/train.py   --training_config configs/beavis.yml   --augment_clips   --log_level INFO
```

### 5.4 All-in-one (generate → augment → train → convert)
```bash
python openWakeWord/openwakeword/train.py   --training_config configs/beavis.yml   --generate_clips   --augment_clips   --train_model   --convert_to_tflite   --overwrite   --log_level INFO
```

---

## 6. Parallelism / performance

### Piper generation workers
Piper generation usually benefits from multiprocessing.

Common knobs:
- `PIPER_WORKERS=16` (example)
- `PIPER_CUDA=1` (only if your Piper build supports GPU)

Example:
```bash
PIPER_WORKERS=16 python openWakeWord/openwakeword/train.py   --training_config configs/beavis.yml   --generate_clips   --log_level INFO
```

### Feature computation multi-process
If your training script supports feature/augment worker variables, prefer:
- `AUGMENT_WORKERS=16`
- `FEATURE_WORKERS=16`

If not supported, the fallback is OS-level parallelism: run separate phases (generate/augment/train) as separate commands and tune each phase.

---

## 7. Troubleshooting (common issues & fixes)

### A) Missing internal models (`melspectrogram.onnx` / embedding)
**Symptom**
- `NoSuchFile: ... melspectrogram.onnx failed. File doesn't exist`

**Fix**
```bash
python -c "import openwakeword.utils as u; u.download_models(['_features_only_'])"
```

---

### B) Piper not found
**Symptom**
- `[Errno 2] No such file or directory: 'piper'`

**Fix options**
1) Install Piper and ensure it is in PATH, or  
2) Set absolute path in config (`piper_executable`), or  
3) Export env var:
```bash
export PIPER_EXE="/full/path/to/piper"
```

---

### C) Piper fails with `ModuleNotFoundError: pathvalidate`
**Symptom**
- Piper CLI crashes: `No module named 'pathvalidate'`

**Fix**
```bash
pip install pathvalidate
```

---

### D) `ModuleNotFoundError: dp` (phonemizer / adversarial texts)
**Symptom**
- `ModuleNotFoundError: No module named 'dp'`

**Fix**
Install the phonemizer dependency used by the pipeline. If this repo provides a patch or extra requirement, install it (example):
```bash
pip install -r requirements/py310.txt
# or, if separated:
pip install deep-phonemizer
```

If your pipeline allows disabling adversarial text generation, you can also disable it in config as a workaround.

---

### E) `pyaudio` install fails (`portaudio.h` missing)
**Symptom**
- `fatal error: portaudio.h: No such file or directory`

**Fix (Ubuntu/Debian)**
```bash
sudo apt update
sudo apt install -y portaudio19-dev python3-dev build-essential
pip install pyaudio
```

If you only need offline testing (not live mic), you can skip `pyaudio`.

---

### F) `torchaudio` backend issues
**Symptom**
- `AttributeError: module 'torchaudio' has no attribute ...`
- backend warnings

**Fix**
Ensure `torch` and `torchaudio` versions match. Reinstall them together using the official PyTorch install command for your platform/CUDA.

---

### G) Feature `.npy` not found (negative_features_test.npy missing)
**Symptom**
- `FileNotFoundError: ... negative_features_test.npy`

**Cause**
You ran `--train_model` before generating/augmenting or before features were computed.

**Fix**
Run augmentation/features first:
```bash
python openWakeWord/openwakeword/train.py --training_config configs/beavis.yml --augment_clips --log_level INFO
```
Or run all-in-one with `--overwrite`.

---

## 8. Reproducibility: how to capture requirements (Python + apt)

### Lock Python environment (inside venv)
```bash
python3 -m pip freeze --local > requirements.lock.txt
```

### Capture system packages (Ubuntu/Debian)
Create `apt.txt`:
```bash
cat > apt.txt <<'EOF'
build-essential
python3-dev
portaudio19-dev
EOF
```

Optionally export the currently installed apt packages:
```bash
apt-mark showmanual > apt.manual.txt
```

---

## 9. What to commit (and what NOT to commit)

✅ Commit:
- `configs/`
- `openWakeWord/` (code only)
- `tools/`
- `scripts/`
- `requirements/`

❌ Do NOT commit:
- `data/` (datasets, voices, large `.npy`)
- `training_results/` (models, generated clips, logs)

---

# Русский

## 1. Что это за репозиторий (и что это не)

Это репозиторий-скелет для обучения wake-word модели на **openWakeWord**, сделанный так, чтобы:
- одинаково работал на **Linux/Windows**
- не ломался из‑за путей
- не тянул в git гигабайты данных

### Внутри есть
- облегчённая копия `openWakeWord/` (код + `train.py`)
- минимальный генератор клипов через Piper (`tools/piper_sample_generator/`)
- переносимые конфиги в `configs/`
- правило: **код/конфиги в git**, **данные/результаты — вне git**

### Внутри НЕТ
- MUSAN, ACAV100M `.npy`, твои `.wav`
- Piper голоса (`*.onnx` + `*.onnx.json`)
- `training_results/` (модели/фичи/логи)

Это кладётся локально в `data/` и `training_results/`.

---

## 2. Структура папок — что где хранится и зачем

### Дерево проекта (с пояснениями)
```
repo/
├── configs/                         # конфиги обучения (переносимые)
├── openWakeWord/                    # openWakeWord внутри репо
│   └── openwakeword/
│       └── resources/models/        # внутренние модели OWw (скачиваются 1 раз)
├── tools/piper_sample_generator/    # генерация клипов через Piper
├── scripts/                         # утилиты (doctor)
├── requirements/                    # списки зависимостей
├── data/                            # НЕ в git: всё тяжёлое и локальное
└── training_results/                # НЕ в git: результат пайплайна
```

### Что именно класть в папки

#### `configs/`
- **Зачем:** настройки обучения (пути, параметры генерации/аугментации/тренировки).
- **Что хранить:** только `.yml`.
- **Проблемы:** абсолютные пути и “сломанная переносимость” → держи пути относительными.

#### `openWakeWord/openwakeword/resources/models/`
- **Зачем:** внутренние ONNX модели, нужные для извлечения признаков (mel + embedding, иногда VAD).
- **Как появляется:** скачивается один раз командой:
  ```bash
  python -c "import openwakeword.utils as u; u.download_models(['_features_only_'])"
  ```
- **Проблема:** если не скачал — будет `melspectrogram.onnx not found`.

#### `tools/piper_sample_generator/`
- **Зачем:** синтетически увеличить позитивы через TTS (Piper).
- **Проблемы:** не найден `piper`, нет `.onnx.json`, не хватает зависимостей Piper (`pathvalidate` и т.п.).

#### `data/` (НЕ коммитить)
- **Зачем:** датасеты, огромные фичи, модели Piper, твои записи.
- **Что должно быть:**
  - `data/my_data/beavis/*.wav` (позитивы)
  - `data/backgrounds/musan/...` (фоны)
  - `data/features/*.npy` (ACAV + validation)
  - `data/piper/voices/*` (голоса Piper, если нужно)

#### `training_results/` (НЕ коммитить)
- **Зачем:** результаты обучения и промежуточные артефакты:
  сгенерированные клипы, аугментация, `.npy` фичи, `*.onnx`, логи.
- **Проблемы:** “частичный прогон” оставляет мусор → используй `--overwrite`, когда нужен чистый rebuild.

---

## 3. Что обязательно для обучения

### Обязательное
1) Позитивы:
```
data/my_data/beavis/*.wav
```

2) MUSAN:
```
data/backgrounds/musan/noise/
data/backgrounds/musan/speech/
data/backgrounds/musan/music/
```

3) ACAV `.npy`:
```
data/features/openwakeword_features_ACAV100M_2000_hrs_16bit.npy
```

### Рекомендуется
4) validation фичи:
```
data/features/validation_set_features.npy
```

### Опционально (для `--generate_clips`)
5) Piper голоса:
```
data/piper/voices/*.onnx
data/piper/voices/*.onnx.json
```

---

## 4. Установка (Python 3.10)

### 4.1 Создание venv

**Linux**
```bash
python3.10 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip wheel setuptools
```

> Если `python` не существует (Ubuntu), используй `python3` или поставь:
> `sudo apt install -y python-is-python3`

**Windows**
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip wheel setuptools
```

### 4.2 Ставим PyTorch отдельно
Ставь по официальному селектору:
- https://pytorch.org/get-started/locally/

Проверка:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 4.3 Зависимости + openWakeWord
```bash
pip install -r requirements/py310.txt
pip install -e ./openWakeWord
```

### 4.4 Скачай внутренние модели openWakeWord (1 раз)
```bash
python -c "import openwakeword.utils as u; u.download_models(['_features_only_'])"
```

---

## 5. Запуск пайплайна

### 5.1 Обучение ONNX (рекомендовано)
```bash
python openWakeWord/openwakeword/train.py   --training_config configs/beavis.yml   --train_model   --log_level INFO
```

### 5.2 Генерация клипов Piper
```bash
python openWakeWord/openwakeword/train.py   --training_config configs/beavis.yml   --generate_clips   --log_level INFO
```

### 5.3 Аугментация + создание фичей
```bash
python openWakeWord/openwakeword/train.py   --training_config configs/beavis.yml   --augment_clips   --log_level INFO
```

### 5.4 Всё одной командой
```bash
python openWakeWord/openwakeword/train.py   --training_config configs/beavis.yml   --generate_clips   --augment_clips   --train_model   --convert_to_tflite   --overwrite   --log_level INFO
```

---

## 6. Производительность и параллельность

### Piper (процессы/воркеры)
Обычно ускоряется за счёт multiprocessing.

Пример:
```bash
PIPER_WORKERS=16 python openWakeWord/openwakeword/train.py   --training_config configs/beavis.yml   --generate_clips   --log_level INFO
```

### Фичи / аугментация (многопоток)
Если в скрипте предусмотрены переменные/параметры воркеров, используй:
- `AUGMENT_WORKERS=16`
- `FEATURE_WORKERS=16`

Если нет — запускай этапы отдельными командами и тюнь каждый этап отдельно.

---

## 7. Типовые проблемы и решения

### A) Нет внутренних моделей (`melspectrogram.onnx`)
**Симптом:** `NoSuchFile ... melspectrogram.onnx`

**Решение:**
```bash
python -c "import openwakeword.utils as u; u.download_models(['_features_only_'])"
```

### B) `piper` не найден
**Симптом:** `[Errno 2] No such file or directory: 'piper'`

**Решения:**
- поставить `piper` в PATH, или
- указать абсолютный путь в `piper_executable`, или
- экспортировать:
```bash
export PIPER_EXE="/полный/путь/к/piper"
```

### C) Piper падает из-за `pathvalidate`
**Симптом:** `No module named 'pathvalidate'`

**Решение:**
```bash
pip install pathvalidate
```

### D) `No module named dp`
**Симптом:** `ModuleNotFoundError: dp`

**Решение:**
Поставь зависимость phonemizer (в зависимости от того, что использует твой пайплайн):
```bash
pip install deep-phonemizer
```
Если есть опция отключить adversarial texts — можно временно отключить в конфиге.

### E) `pyaudio` не ставится (`portaudio.h`)
**Симптом:** `fatal error: portaudio.h: No such file`

**Решение:**
```bash
sudo apt update
sudo apt install -y portaudio19-dev python3-dev build-essential
pip install pyaudio
```

### F) Проблемы `torchaudio`
**Симптом:** backend ошибки/варнинги/attribute errors

**Решение:** переустановить `torch` + `torchaudio` одной командой с официального PyTorch селектора.

### G) Нет `.npy` фичей (например `negative_features_test.npy`)
**Симптом:** `FileNotFoundError: negative_features_test.npy`

**Причина:** запущен `--train_model` до этапа создания фичей.

**Решение:**
```bash
python openWakeWord/openwakeword/train.py --training_config configs/beavis.yml --augment_clips --log_level INFO
```
или end-to-end с `--overwrite`.

---

## 8. Как собрать “полный requirements” (воспроизводимость)

### Python lock (внутри venv)
```bash
python3 -m pip freeze --local > requirements.lock.txt
```

### Системные пакеты (Ubuntu/Debian)
```bash
cat > apt.txt <<'EOF'
build-essential
python3-dev
portaudio19-dev
EOF
```

Опционально:
```bash
apt-mark showmanual > apt.manual.txt
```

---

## 9. Что коммитить на GitHub

✅ Коммитить:
- `configs/`, `openWakeWord/` (код), `tools/`, `scripts/`, `requirements/`

❌ Не коммитить:
- `data/`, `training_results/`

---

### License / Credits
- This workspace wraps a vendored copy of **openWakeWord** (credits to upstream authors).
- Datasets and large features remain the responsibility of the user and should be obtained from their original sources.
