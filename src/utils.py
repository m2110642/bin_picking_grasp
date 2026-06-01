import json
import os
from pathlib import Path

import numpy as np
import yaml


def load_config(config_path: str):
    """
    Загружает YAML-конфигурацию проекта
    """
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def resolve_path(project_root: str, path_from_config: str):
    path = Path(path_from_config)

    if path.is_absolute():
        return str(path)

    return str(Path(project_root) / path)


def ensure_dir(path: str):
    """
    Создаёт папку, если её ещё нет
    """
    os.makedirs(path, exist_ok=True)


def save_json(data, path: str):
    """
    Сохраняет данные JSON
    """
    ensure_dir(os.path.dirname(path))

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_cloud(cloud_path: str):
    """
    Загружает cloud .npy
    """
    if not os.path.exists(cloud_path):
        raise FileNotFoundError(f"Cloud файл не найден {cloud_path}")

    cloud = np.load(cloud_path)

    if cloud.ndim != 3 or cloud.shape[2] != 3:
        raise ValueError(
            f"Некорректный формат cloud: {cloud_path}. "
            f"Ожидался массив H x W x 3, но получено: {cloud.shape}"
        )

    return cloud


def find_rgb_cloud_pairs(images_dir: str, cloud_dir: str):
    """
    Находит пары RGB + cloud по совпадению имен файлов
    """
    allowed_ext = (".jpg", ".jpeg", ".png")

    image_files = [
        file for file in os.listdir(images_dir)
        if file.lower().endswith(allowed_ext)
    ]

    pairs = []

    for image_file in sorted(image_files):
        base_name = os.path.splitext(image_file)[0]
        cloud_file = base_name + ".npy"

        image_path = os.path.join(images_dir, image_file)
        cloud_path = os.path.join(cloud_dir, cloud_file)

        if os.path.exists(cloud_path):
            pairs.append({
                "base_name": base_name,
                "image_file": image_file,
                "cloud_file": cloud_file,
                "image_path": image_path,
                "cloud_path": cloud_path
            })

    return pairs