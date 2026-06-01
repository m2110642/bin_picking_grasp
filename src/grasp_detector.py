import math
import cv2
import numpy as np

from src.object_prioritizer import get_centroid, get_valid_cloud_mask, minmax_normalize


def is_valid_point(point):
    """
    Проверка валидности одной 3д точки
    """
    if point is None:
        return False

    point = np.asarray(point)

    if point.shape[0] != 3:
        return False

    if not np.isfinite(point).all():
        return False

    if np.all(np.isclose(point, 0.0)):
        return False

    return True


def compute_local_normal(valid_points):
    """
    Оценивает нормаль локальной поверхности по валидным точкам

    Используется SVD/PCA
    два направления максимального разброса лежат вдоль поверхности
    направление минимального разброса считается нормалью
    """
    if len(valid_points) < 6:
        return None, np.nan

    pts = valid_points.astype(np.float64)
    pts_centered = pts - pts.mean(axis=0, keepdims=True)

    try:
        _, _, vh = np.linalg.svd(pts_centered, full_matrices=False)
        normal = vh[-1, :]
    except Exception:
        return None, np.nan

    norm = np.linalg.norm(normal)

    if norm < 1e-8:
        return None, np.nan

    normal = normal / norm

    normal_score = abs(float(normal[2]))

    return normal, normal_score


def compute_candidate_raw_metrics(
    x: int,
    y: int,
    mask: np.ndarray,
    cloud: np.ndarray,
    distance_map: np.ndarray,
    half_window: int,
    min_valid_ratio: float
):
    """
    Считает сырые признаки точки кандидата
    """
    h, w = mask.shape

    if x < half_window or y < half_window:
        return None

    if x >= w - half_window or y >= h - half_window:
        return None

    center_point = cloud[y, x]

    if not is_valid_point(center_point):
        return None

    y1 = y - half_window
    y2 = y + half_window + 1
    x1 = x - half_window
    x2 = x + half_window + 1

    mask_window = mask[y1:y2, x1:x2]
    cloud_window = cloud[y1:y2, x1:x2, :]

    object_points = cloud_window[mask_window > 0]

    if len(object_points) == 0:
        return None

    valid_mask = get_valid_cloud_mask(object_points)
    valid_points = object_points[valid_mask]

    valid_ratio = len(valid_points) / max(len(object_points), 1)

    if valid_ratio < min_valid_ratio:
        return None

    if len(valid_points) < 6:
        return None

    z_values = valid_points[:, 2]
    z_values = z_values[np.isfinite(z_values)]

    if len(z_values) < 6:
        return None

    z_std = float(np.nanstd(z_values))
    z_range = float(np.nanmax(z_values) - np.nanmin(z_values))

    normal, normal_score = compute_local_normal(valid_points)

    edge_distance = float(distance_map[y, x])

    obj_cx, obj_cy = get_centroid(mask)
    dist_to_centroid = math.sqrt((x - obj_cx) ** 2 + (y - obj_cy) ** 2)

    max_obj_dist = math.sqrt(mask.shape[0] ** 2 + mask.shape[1] ** 2)
    centrality_raw = 1.0 - dist_to_centroid / max(max_obj_dist, 1.0)
    centrality_raw = float(np.clip(centrality_raw, 0.0, 1.0))

    return {
        "x": int(x),
        "y": int(y),
        "point_3d": center_point.tolist(),
        "valid_cloud_ratio": float(valid_ratio),
        "z_std": z_std,
        "z_range": z_range,
        "normal_score": float(normal_score) if np.isfinite(normal_score) else 0.0,
        "edge_distance_px": edge_distance,
        "centrality_raw": centrality_raw,
        "normal": normal.tolist() if normal is not None else None
    }


def find_grasp_point(
    mask: np.ndarray,
    cloud: np.ndarray,
    weights: dict,
    window_size: int = 21,
    candidate_stride: int = 4,
    max_candidates: int = 1800,
    min_valid_ratio: float = 0.55,
    min_edge_distance_px: float = 4
):
    """
    Ищет оптимальную точку захвата внутри маски объекта.

    Возвращает:
    best: лучший кандидат
    candidates: список всех оценённых кандидатов, отсортированный по grasp_score
    heatmap: карта качества точек захвата
    """
    mask_uint8 = mask.astype(np.uint8)

    if mask_uint8.sum() == 0:
        return None, None, None

    half_window = window_size // 2
    distance_map = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)

    valid_candidate_coords = []

    ys, xs = np.where(mask_uint8 > 0)

    for y, x in zip(ys, xs):
        if y % candidate_stride != 0 or x % candidate_stride != 0:
            continue

        if distance_map[y, x] < min_edge_distance_px:
            continue

        if not is_valid_point(cloud[y, x]):
            continue

        valid_candidate_coords.append((x, y))

    if len(valid_candidate_coords) == 0:
        return None, None, None

    if len(valid_candidate_coords) > max_candidates:
        coords_with_dist = [
            (x, y, distance_map[y, x])
            for x, y in valid_candidate_coords
        ]
        coords_with_dist = sorted(coords_with_dist, key=lambda t: t[2], reverse=True)
        coords_with_dist = coords_with_dist[:max_candidates]
        valid_candidate_coords = [(x, y) for x, y, _ in coords_with_dist]

    raw_candidates = []

    for x, y in valid_candidate_coords:
        raw = compute_candidate_raw_metrics(
            x=x,
            y=y,
            mask=mask_uint8,
            cloud=cloud,
            distance_map=distance_map,
            half_window=half_window,
            min_valid_ratio=min_valid_ratio
        )

        if raw is not None:
            raw_candidates.append(raw)

    if len(raw_candidates) == 0:
        return None, None, None

    valid_ratios = [c["valid_cloud_ratio"] for c in raw_candidates]
    z_stds = [c["z_std"] for c in raw_candidates]
    z_ranges = [c["z_range"] for c in raw_candidates]
    edge_distances = [c["edge_distance_px"] for c in raw_candidates]
    normal_scores = [c["normal_score"] for c in raw_candidates]
    centralities = [c["centrality_raw"] for c in raw_candidates]

    valid_scores = minmax_normalize(valid_ratios, higher_is_better=True)

    flatness_from_std = minmax_normalize(z_stds, higher_is_better=False)
    flatness_from_range = minmax_normalize(z_ranges, higher_is_better=False)
    flatness_scores = 0.6 * flatness_from_std + 0.4 * flatness_from_range

    edge_scores = minmax_normalize(edge_distances, higher_is_better=True)
    normal_scores_norm = np.array(normal_scores, dtype=np.float32)
    centrality_scores = minmax_normalize(centralities, higher_is_better=True)

    heatmap = np.zeros(mask_uint8.shape, dtype=np.float32)
    scored_candidates = []

    for i, candidate in enumerate(raw_candidates):
        grasp_score = (
            weights["valid_cloud"] * float(valid_scores[i]) +
            weights["flatness"] * float(flatness_scores[i]) +
            weights["edge_distance"] * float(edge_scores[i]) +
            weights["normal"] * float(normal_scores_norm[i]) +
            weights["centrality"] * float(centrality_scores[i])
        )

        item = dict(candidate)
        item["valid_cloud_score"] = float(valid_scores[i])
        item["flatness_score"] = float(flatness_scores[i])
        item["edge_distance_score"] = float(edge_scores[i])
        item["normal_score_norm"] = float(normal_scores_norm[i])
        item["centrality_score"] = float(centrality_scores[i])
        item["grasp_score"] = float(grasp_score)

        scored_candidates.append(item)
        heatmap[candidate["y"], candidate["x"]] = grasp_score

    scored_candidates = sorted(
        scored_candidates,
        key=lambda item: item["grasp_score"],
        reverse=True
    )

    best = scored_candidates[0]

    heatmap_blur = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=7, sigmaY=7)
    heatmap_blur[mask_uint8 == 0] = 0.0

    if heatmap_blur.max() > 0:
        heatmap_blur = heatmap_blur / heatmap_blur.max()

    return best, scored_candidates, heatmap_blur