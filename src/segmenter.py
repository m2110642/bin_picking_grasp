import cv2
import numpy as np
import torch


def read_rgb_image(image_path: str):
    """
    Читает изображение через opencv и переводит BGR в RGB
    """
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb


def image_to_tensor(image_rgb: np.ndarray, device: str = "cpu"):
    """
    Преобразует RGB изображение H×W×3 в torch tensor 3×H×W
    """
    image_tensor = torch.as_tensor(
        image_rgb / 255.0,
        dtype=torch.float32
    ).permute(2, 0, 1)

    return image_tensor.to(device)


def run_segmentation(
    model,
    image_rgb: np.ndarray,
    device: str = "cpu",
    confidence_threshold: float = 0.5,
    mask_threshold: float = 0.5
):
    """
    Запускает Mask R-CNN на одном RGB изображении и возвращает:
    masks: list[np.ndarray] бинарные маски H×W
    boxes: np.ndarray Nx4
    scores: np.ndarray N
    """

    image_tensor = image_to_tensor(image_rgb, device=device)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    scores = prediction["scores"].detach().cpu().numpy()
    keep = scores >= confidence_threshold

    filtered_scores = scores[keep]
    filtered_boxes = prediction["boxes"].detach().cpu().numpy()[keep]

    filtered_masks = (
        prediction["masks"].detach().cpu().numpy()[keep, 0] > mask_threshold
    ).astype(np.uint8)

    masks = [filtered_masks[i] for i in range(len(filtered_masks))]

    return masks, filtered_boxes, filtered_scores