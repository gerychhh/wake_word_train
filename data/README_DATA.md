# Data folders (NOT tracked in git)

Сюда кладём тяжёлые данные (датасеты/фичи/голоса), чтобы репозиторий был лёгким.

## 1) Позитивы + FP-валидация (совместимость с `my_data`)
- **Позитивы (твои записи):** `data/my_data/beavis/*.wav`
- **FP/h валидация (рекомендовано):** `data/features/validation_set_features.npy`

## 2) MUSAN backgrounds
После распаковки MUSAN должны существовать папки:
- `data/backgrounds/musan/noise/`
- `data/backgrounds/musan/speech/`
- `data/backgrounds/musan/music/`

## 3) Feature .npy (openWakeWord)
Обязательный файл (очень большой, ~17GB):
- `data/features/openwakeword_features_ACAV100M_2000_hrs_16bit.npy`

## 4) Piper voice model (опционально, только для `--generate_clips`)
Положи Piper-голос:
- `data/piper/ru_RU-dmitri-medium.onnx`
- рядом должен быть файл конфигурации: `ru_RU-dmitri-medium.onnx.json`

Сгенерированные клипы пишутся сюда (можно удалять):
- `data/_generated_clips/beavis/`
