import os
import requests
import tqdm
import zipfile

def download(url, filename):
    if os.path.exists(filename): return
    print(f"Загрузка {filename}...")
    r = requests.get(url, stream=True)
    total = int(r.headers.get('content-length', 0))
    with open(filename, 'wb') as f, tqdm.tqdm(total=total, unit='B', unit_scale=True) as bar:
        for data in r.iter_content(chunk_size=1024):
            f.write(data)
            bar.update(len(data))

# Создаем структуру папок
base = "./openWakeWord"
os.makedirs(f"{base}/mit_rirs", exist_ok=True)
os.makedirs(f"{base}/background_clips", exist_ok=True)

# 1. Скачиваем признаки (Самое важное для конфига)
print("--- Загрузка признаков ---")
features = {
    "openwakeword_features_ACAV100M_2000_hrs_16bit.npy": "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
    "validation_set_features.npy": "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"
}
for name, url in features.items():
    download(url, name)

# 2. Скачиваем RIR (Эхо) и Фоны (Шумы) - прямые ссылки на архивы
print("\n--- Загрузка окружения ---")
env_data = {
    f"{base}/mit_rirs.zip": "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/mit_rirs.zip",
    f"{base}/background_clips.zip": "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/background_clips.zip"
}

for path, url in env_data.items():
    download(url, path)
    print(f"Распаковка {path}...")
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(base)
    os.remove(path)

print("\n--- ВСЁ ГОТОВО! Все файлы на месте. ---")