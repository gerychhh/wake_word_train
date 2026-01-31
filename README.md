# beavis-wakeword-trainer (clean, portable)

Это **чистый проект без .git**, с нормальной структурой и портируемыми путями (Windows/Linux).
Ты можешь **полностью снести старую папку**, распаковать этот архив и просто разложить данные в `data/` — после этого обучение запускается как раньше.

Важное:
- Код openWakeWord здесь уже внутри (`openWakeWord/`), но **без git-истории, тестов и ноутбуков**
- Тяжёлые файлы (MUSAN, ACAV100M, Piper, твои wav) **не входят** — их надо положить в `data/`
- Твой конфиг **перенесён 1-в-1** в `configs/beavis.yml`
- Оригинальный Windows-конфиг сохранён как `configs/custom_model_windows_original.yml` (чтобы сравнить)

---

## Структура папок

- `configs/` — YAML конфиги (пути относительные)
- `openWakeWord/` — библиотека + `train.py`
- `tools/piper_sample_generator/` — минимальный генератор клипов (Piper CLI)
- `data/` — **данные (НЕ в git)**, см. `data/README_DATA.md`
- `training_results/` — результаты обучения (`beavis.onnx`, опционально `beavis.tflite`, логи, временные клипы)

---

## Куда класть данные (самое важное)

Открой `data/README_DATA.md`, там коротко, но вот супер-выжимка:

### 1) Твои позитивы
Клади сюда:
- `data/my_data/beavis/*.wav`

### 2) MUSAN (фоны)
Должно получиться:
- `data/backgrounds/musan/noise/`
- `data/backgrounds/musan/speech/`
- `data/backgrounds/musan/music/`

### 3) ACAV100M фичи (обязательно, очень большой файл ~17GB)
- `data/features/openwakeword_features_ACAV100M_2000_hrs_16bit.npy`

### 4) FP/h валидация (рекомендовано)
- `data/features/validation_set_features.npy`

### 5) Piper (опционально, только если используешь `--generate_clips`)
- `data/piper/ru_RU-dmitri-medium.onnx`
- `data/piper/ru_RU-dmitri-medium.onnx.json`

---

## Установка (Python 3.10)

### 0) Создай venv

#### Linux
```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
```

#### Windows
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip wheel setuptools
```

### 1) Установи PyTorch (CPU или CUDA)
PyTorch ставим отдельно, чтобы не словить конфликт версий:
- через официальный селектор: https://pytorch.org/get-started/locally/

Проверка:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 2) Установи остальные зависимости + openWakeWord
```bash
pip install -r requirements/py310.txt
pip install -e ./openWakeWord
```

### 3) Быстрый чек окружения
```bash
python scripts/doctor.py
```

### 2.5) Скачай внутренние модели openWakeWord (нужно 1 раз)
Если видишь ошибку про `melspectrogram.onnx` / `embedding_model.onnx` — значит эти файлы ещё не скачаны.

Выполни:
```bash
python -c "import openwakeword.utils as u; u.download_models(['_features_only_'])"
```

Это скачает **feature-модели** (melspectrogram + embedding) и VAD в:
`openWakeWord/openwakeword/resources/models/`


---

## Запуск обучения

### Базовый вариант (рекомендуется): ONNX
Это даст тебе `training_results/beavis.onnx` (и всё обучение как раньше):
```bash
python openWakeWord/openwakeword/train.py --training_config configs/beavis.yml --train_model --log_level INFO
```

### Генерация/аугментация (если нужно)
```bash
python openWakeWord/openwakeword/train.py --training_config configs/beavis.yml --generate_clips --log_level INFO
python openWakeWord/openwakeword/train.py --training_config configs/beavis.yml --augment_clips --log_level INFO
```

> Генерация использует Piper CLI. Исполняемый файл берётся из `configs/beavis.yml` (`piper_executable`)
> или из переменной окружения `PIPER_EXE`.
> Для ускорения:
> - `PIPER_WORKERS=8`
> - `PIPER_CUDA=1` (если твой Piper поддерживает `--cuda`)

### Опционально: TFLite конвертация
Если тебе реально нужен `beavis.tflite`, поставь доп. зависимости:
```bash
pip install -r requirements/py310-tflite.txt
```

И запускай с флагом:
```bash
python openWakeWord/openwakeword/train.py --training_config configs/beavis.yml --train_model --convert_to_tflite --log_level INFO
```

> Важно: onnx→tflite — капризная часть экосистемы, иногда требует подбора версий.
> Если конвертация не критична — используй ONNX.

---

## GitHub

Репа уже готова к твоему git:
- `.gitignore` скрывает все тяжёлые данные и артефакты
- папки-скелеты сохраняются через `.gitkeep`

Команды:
```bash
git init
git add .
git commit -m "initial clean wakeword trainer"
```

---

## Где настроено “как на Windows”
`configs/beavis.yml` перенесён из твоего `custom_model.yml`, включая:
- `steps: 1750000`
- `max_negative_weight: 900`
- `target_false_positives_per_hour: 0.3`
- `background_paths_duplication_rate: [1, 3, 5]`
- `custom_negative_phrases: ...` (весь список)