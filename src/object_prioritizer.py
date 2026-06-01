import math
import numpy as np


def get_centroid(mask: np.ndarray):
    """
    Возвращает центроид бинарной маски
    """
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return 0.0, 0.0

    return float(xs.mean()), float(ys.mean())


def get_valid_cloud_mask(points: np.ndarray):
    """
    Возвращает маску валидных 3д точек
    Валидная точка:
    - конечная
    - не равна [0, 0, 0]
    """
    if points.size == 0:
        return np.zeros((0,), dtype=bool)

    finite = np.isfinite(points).all(axis=1)

    non_zero = np.logical_not(
        np.logical_and.reduce([
            np.isclose(points[:, 0], 0.0),
            np.isclose(points[:, 1], 0.0),
            np.isclose(points[:, 2], 0.0)
        ])
    )

    return np.logical_and(finite, non_zero)


def minmax_normalize(values, higher_is_better=True):
    """
    Min-max нормализация списка значений в диапазон [0, 1]
    """
    values = np.array(values, dtype=np.float32)

    if len(values) == 0:
        return values

    finite = np.isfinite(values)

    if finite.sum() == 0:
        return np.zeros_like(values, dtype=np.float32)

    v_min = np.nanmin(values[finite])
    v_max = np.nanmax(values[finite])

    if abs(v_max - v_min) < 1e-8:
        return np.ones_like(values, dtype=np.float32)

    values_filled = values.copy()
    values_filled[~finite] = v_max if higher_is_better else v_min

    norm = (values_filled - v_min) / (v_max - v_min)

    if higher_is_better:
        return norm

    return 1.0 - norm


def compute_area_score(mask: np.ndarray, image_h: int, image_w: int):
    """
    Нормированная площадь маски относительно изображения
    """
    image_area = float(image_h * image_w)

    if image_area <= 0:
        return 0.0

    return float(mask.sum()) / image_area


def compute_center_score(mask: np.ndarray, image_h: int, image_w: int):
    """
    Оценка центральности объекта
    Чем ближе центроид объекта к центру кадра, тем выше score-оценка
    """
    cx, cy = get_centroid(mask)

    image_cx = image_w / 2.0
    image_cy = image_h / 2.0

    distance = math.sqrt((cx - image_cx) ** 2 + (cy - image_cy) ** 2)
    max_distance = math.sqrt((image_w / 2.0) ** 2 + (image_h / 2.0) ** 2)

    if max_distance <= 0:
        return 0.0

    normalized_distance = distance / max_distance
    center_score = 1.0 - normalized_distance

    return float(np.clip(center_score, 0.0, 1.0))


def compute_overlap_score(mask: np.ndarray, all_masks, idx: int):
    """
    Оценка неперекрытости объекта
    Чем меньше маска перекрывается с другими, тем выше score
    """
    mask_bool = mask.astype(bool)

    if mask_bool.sum() == 0:
        return 0.0

    others = np.zeros_like(mask_bool, dtype=bool)

    for j, other_mask in enumerate(all_masks):
        if j == idx:
            continue
        others = np.logical_or(others, other_mask.astype(bool))

    overlap_ratio = np.logical_and(mask_bool, others).sum() / mask_bool.sum()
    overlap_score = 1.0 - overlap_ratio

    return float(np.clip(overlap_score, 0.0, 1.0))


def compute_raw_object_features(
    mask: np.ndarray,
    score: float,
    cloud: np.ndarray,
    all_masks,
    idx: int,
    z_mode: str = "smaller_z_is_higher"
):
    """
    Считает сырые признаки объекта
    """
    image_h, image_w = mask.shape[:2]

    confidence_score = float(score)
    area_raw = compute_area_score(mask, image_h, image_w)
    center_score = compute_center_score(mask, image_h, image_w)
    overlap_score = compute_overlap_score(mask, all_masks, idx)

    points = cloud[mask > 0]

    if len(points) == 0:
        valid_cloud_ratio = 0.0
        top_z = np.nan
        mean_z = np.nan
    else:
        valid_mask = get_valid_cloud_mask(points)
        valid_points = points[valid_mask]

        valid_cloud_ratio = float(len(valid_points) / max(len(points), 1))

        if len(valid_points) == 0:
            top_z = np.nan
            mean_z = np.nan
        else:
            z_values = valid_points[:, 2]
            z_values = z_values[np.isfinite(z_values)]

            if len(z_values) == 0:
                top_z = np.nan
                mean_z = np.nan
            else:
                if z_mode == "smaller_z_is_higher":
                    top_z = float(np.nanpercentile(z_values, 10))
                else:
                    top_z = float(np.nanpercentile(z_values, 90))

                mean_z = float(np.nanmean(z_values))

    cx, cy = get_centroid(mask)

    return {
        "object_id": int(idx),
        "confidence_score": confidence_score,
        "area_raw": area_raw,
        "center_score": center_score,
        "overlap_score": overlap_score,
        "valid_cloud_score": valid_cloud_ratio,
        "top_z_raw": top_z,
        "mean_z_raw": mean_z,
        "centroid_x": cx,
        "centroid_y": cy,
        "object_pixels": int(mask.sum())
    }


def build_features_for_scene(
    masks,
    scores,
    cloud: np.ndarray,
    z_mode: str = "smaller_z_is_higher"
):
    """
    Считает признаки для всех объектов сцены
    """
    raw_features = []

    for idx in range(len(masks)):
        raw = compute_raw_object_features(
            mask=masks[idx],
            score=scores[idx],
            cloud=cloud,
            all_masks=masks,
            idx=idx,
            z_mode=z_mode
        )
        raw_features.append(raw)

    if len(raw_features) == 0:
        return []

    area_values = [f["area_raw"] for f in raw_features]
    area_scores = minmax_normalize(area_values, higher_is_better=True)

    z_values = [f["top_z_raw"] for f in raw_features]
    z_array = np.array(z_values, dtype=np.float32)
    finite_z = np.isfinite(z_array)

    if finite_z.sum() == 0:
        height_scores = np.zeros_like(z_array, dtype=np.float32)
    else:
        z_filled = z_array.copy()
        z_filled[~finite_z] = np.nanmax(z_array[finite_z])

        if z_mode == "smaller_z_is_higher":
            height_scores = minmax_normalize(z_filled, higher_is_better=False)
        else:
            height_scores = minmax_normalize(z_filled, higher_is_better=True)

    features = []

    for i, f in enumerate(raw_features):
        item = {
            "object_id": f["object_id"],
            "confidence_score": float(np.clip(f["confidence_score"], 0.0, 1.0)),
            "area_score": float(np.clip(area_scores[i], 0.0, 1.0)),
            "center_score": float(np.clip(f["center_score"], 0.0, 1.0)),
            "height_score": float(np.clip(height_scores[i], 0.0, 1.0)),
            "valid_cloud_score": float(np.clip(f["valid_cloud_score"], 0.0, 1.0)),
            "overlap_score": float(np.clip(f["overlap_score"], 0.0, 1.0)),
            "top_z_raw": f["top_z_raw"],
            "mean_z_raw": f["mean_z_raw"],
            "centroid_x": f["centroid_x"],
            "centroid_y": f["centroid_y"],
            "object_pixels": f["object_pixels"]
        }
        features.append(item)

    return features


def compute_priority(feature: dict, weights: dict):
    """
    Считает priority score объекта.
    """
    return float(
        weights["confidence"] * feature["confidence_score"] +
        weights["area"] * feature["area_score"] +
        weights["center"] * feature["center_score"] +
        weights["height"] * feature["height_score"] +
        weights["valid_cloud"] * feature["valid_cloud_score"] +
        weights["overlap"] * feature["overlap_score"]
    )


def rank_objects(features, weights: dict):
    """
    Сортирует объекты по приоритету
    """
    ranked = []

    for feature in features:
        item = dict(feature)
        item["priority"] = compute_priority(feature, weights)
        ranked.append(item)

    ranked = sorted(ranked, key=lambda x: x["priority"], reverse=True)

    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank

    return ranked