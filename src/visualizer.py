import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_expanded_bbox_from_mask(mask: np.ndarray, image_h: int, image_w: int, pad: int = 60):
    """
    Возвращает bbox вокруг маски.
    Формат: x1, y1, x2, y2.
    """
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, image_w - 1, image_h - 1

    x1 = max(int(xs.min()) - pad, 0)
    y1 = max(int(ys.min()) - pad, 0)
    x2 = min(int(xs.max()) + pad, image_w - 1)
    y2 = min(int(ys.max()) + pad, image_h - 1)

    return x1, y1, x2, y2


def crop_array(arr: np.ndarray, bbox):
    """
    Обрезает массив по bbox
    bbox x1, y1, x2, y2
    """
    x1, y1, x2, y2 = bbox
    return arr[y1:y2 + 1, x1:x2 + 1]


def draw_pipeline_result(
    image_rgb: np.ndarray,
    masks,
    ranked_objects,
    best_obj: dict,
    grasp: dict,
    heatmap: np.ndarray,
    cloud: np.ndarray,
    save_path: str,
    crop_padding: int = 60
):
    """
    Итоговая визуализация
    1 Общая сцена с приоритизацией
    2 Увеличенный выбранный объект + точка захвата
    3 Увеличенная heatmap качества точек захвата
    4 Z-канал только выбранного объекта + точка захвата
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    image_h, image_w = image_rgb.shape[:2]
    best_mask = masks[best_obj["object_id"]].astype(np.uint8)

    crop_bbox = get_expanded_bbox_from_mask(
        mask=best_mask,
        image_h=image_h,
        image_w=image_w,
        pad=crop_padding
    )

    x1, y1, x2, y2 = crop_bbox

    image_crop = crop_array(image_rgb, crop_bbox)
    mask_crop = crop_array(best_mask, crop_bbox)
    heatmap_crop = crop_array(heatmap, crop_bbox)

    grasp_x_crop = grasp["x"] - x1
    grasp_y_crop = grasp["y"] - y1

    z_full = cloud[:, :, 2].copy()
    z_full[~np.isfinite(z_full)] = np.nan

    z_object_only = np.full_like(z_full, np.nan, dtype=np.float32)
    z_object_only[best_mask > 0] = z_full[best_mask > 0]

    z_object_crop = crop_array(z_object_only, crop_bbox)

    valid_z = z_object_crop[np.isfinite(z_object_crop)]

    if len(valid_z) > 0:
        z_vmin = np.nanpercentile(valid_z, 2)
        z_vmax = np.nanpercentile(valid_z, 98)

        if abs(z_vmax - z_vmin) < 1e-8:
            z_vmin = np.nanmin(valid_z)
            z_vmax = np.nanmax(valid_z)
    else:
        z_vmin = None
        z_vmax = None

    fig, axes = plt.subplots(1, 4, figsize=(30, 8))


    # 1. Общая сцена с рангами

    axes[0].imshow(image_rgb)
    axes[0].set_title("Приоритизация объектов", fontsize=14, fontweight="bold")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    for obj in ranked_objects:
        obj_id = obj["object_id"]
        mask = masks[obj_id].astype(np.uint8)

        if obj["rank"] == 1:
            color = np.array([0.0, 0.85, 0.2])
            alpha = 0.45
            linewidth = 3
        else:
            color = np.array([1.0, 0.65, 0.0])
            alpha = 0.18
            linewidth = 1.2

        overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
        overlay[:, :, 0] = color[0]
        overlay[:, :, 1] = color[1]
        overlay[:, :, 2] = color[2]
        overlay[:, :, 3] = mask.astype(np.float32) * alpha
        axes[0].imshow(overlay)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            cnt = cnt.squeeze()
            if len(cnt.shape) == 2 and len(cnt) > 2:
                axes[0].plot(
                    np.append(cnt[:, 0], cnt[0, 0]),
                    np.append(cnt[:, 1], cnt[0, 1]),
                    color=color,
                    linewidth=linewidth
                )

        if obj["rank"] <= 5:
            axes[0].text(
                obj["centroid_x"],
                obj["centroid_y"],
                f"#{obj['rank']}",
                color="white",
                fontsize=13 if obj["rank"] == 1 else 10,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor=color,
                    alpha=0.95
                )
            )

    rect = plt.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=2.5,
        edgecolor="red",
        facecolor="none",
        linestyle="--"
    )
    axes[0].add_patch(rect)

    # 2. Увеличенный выбранный объект + точка


    axes[1].imshow(image_crop)
    axes[1].set_title("Увеличенный объект rank #1", fontsize=14, fontweight="bold")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    overlay = np.zeros((mask_crop.shape[0], mask_crop.shape[1], 4), dtype=np.float32)
    overlay[:, :, 1] = 0.85
    overlay[:, :, 3] = mask_crop.astype(np.float32) * 0.42
    axes[1].imshow(overlay)

    contours, _ = cv2.findContours(
        mask_crop,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        cnt = cnt.squeeze()
        if len(cnt.shape) == 2 and len(cnt) > 2:
            axes[1].plot(
                np.append(cnt[:, 0], cnt[0, 0]),
                np.append(cnt[:, 1], cnt[0, 1]),
                color="lime",
                linewidth=3
            )

    axes[1].scatter(
        [grasp_x_crop],
        [grasp_y_crop],
        s=250,
        c="red",
        marker="x",
        linewidths=5
    )

    axes[1].text(
        grasp_x_crop + 8,
        grasp_y_crop - 8,
        f"score={grasp['grasp_score']:.3f}",
        color="white",
        fontsize=12,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="red",
            alpha=0.85
        )
    )


    # 3 Heatmap качества точек


    axes[2].imshow(image_crop)
    axes[2].imshow(heatmap_crop, cmap="jet", alpha=0.62)
    axes[2].scatter(
        [grasp_x_crop],
        [grasp_y_crop],
        s=250,
        c="white",
        marker="x",
        linewidths=5
    )
    axes[2].set_title("Heatmap качества точек захвата", fontsize=14, fontweight="bold")
    axes[2].set_xticks([])
    axes[2].set_yticks([])


    # 4 Z-канал только выбранного объекта


    axes[3].set_title("Z-канал выбранного объекта", fontsize=14, fontweight="bold")
    axes[3].set_xticks([])
    axes[3].set_yticks([])

    axes[3].imshow(image_crop, alpha=0.25)

    if len(valid_z) > 0:
        axes[3].imshow(
            z_object_crop,
            cmap="viridis",
            alpha=0.9,
            vmin=z_vmin,
            vmax=z_vmax
        )
    else:
        axes[3].imshow(z_object_crop, cmap="viridis", alpha=0.9)

    axes[3].scatter(
        [grasp_x_crop],
        [grasp_y_crop],
        s=250,
        c="red",
        marker="x",
        linewidths=5
    )

    info = (
        f"Object ID: {best_obj['object_id']} | "
        f"Priority: {best_obj['priority']:.3f} | "
        f"Grasp score: {grasp['grasp_score']:.3f} | "
        f"2D point: ({grasp['x']}, {grasp['y']}) | "
        f"3D point: [{grasp['point_3d'][0]:.3f}, {grasp['point_3d'][1]:.3f}, {grasp['point_3d'][2]:.3f}] | "
        f"Valid: {grasp['valid_cloud_ratio']:.2f} | "
        f"Flatness: {grasp['flatness_score']:.2f} | "
        f"Edge: {grasp['edge_distance_score']:.2f} | "
        f"Normal: {grasp['normal_score_norm']:.2f}"
    )

    fig.text(
        0.5,
        0.025,
        info,
        ha="center",
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.55",
            facecolor="white",
            edgecolor="#CCCCCC",
            alpha=0.95
        )
    )

    plt.tight_layout(rect=[0, 0.09, 1, 1])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()