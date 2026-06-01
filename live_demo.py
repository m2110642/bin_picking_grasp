import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from src.model_loader import load_mask_rcnn_model
from src.realsense_camera import RealSenseCamera
from src.segmenter import run_segmentation
from src.object_prioritizer import build_features_for_scene, rank_objects
from src.grasp_detector import find_grasp_point
from src.utils import load_config, resolve_path


def draw_live_view(image_rgb, masks, ranked_objects, best_obj, grasp):
    frame = image_rgb.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if best_obj is None or grasp is None:
        cv2.putText(
            frame,
            "No grasp point",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )
        return frame

    best_id = best_obj["object_id"]

    for obj in ranked_objects:
        obj_id = obj["object_id"]
        mask = masks[obj_id].astype(np.uint8)

        if obj_id == best_id:
            color = (0, 255, 0)
            thickness = 3
        else:
            color = (0, 180, 255)
            thickness = 1

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(frame, contours, -1, color, thickness)

        if obj["rank"] <= 5:
            cx = int(obj["centroid_x"])
            cy = int(obj["centroid_y"])

            cv2.putText(
                frame,
                f"#{obj['rank']}",
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    x = int(grasp["x"])
    y = int(grasp["y"])

    cv2.drawMarker(
        frame,
        (x, y),
        (0, 0, 255),
        markerType=cv2.MARKER_TILTED_CROSS,
        markerSize=28,
        thickness=3
    )

    text_1 = f"obj={best_obj['object_id']} priority={best_obj['priority']:.3f}"
    text_2 = f"grasp=({x},{y}) score={grasp['grasp_score']:.3f}"

    cv2.putText(
        frame,
        text_1,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        text_2,
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return frame


def main():
    root = Path(__file__).resolve().parent
    config = load_config(str(root / "config.yaml"))

    model_path = resolve_path(str(root), config["paths"]["model_path"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("live demo")
    print("device:", device)
    print("model:", model_path)
    print("press q to exit")

    model = load_mask_rcnn_model(
        model_path=model_path,
        num_classes=config["model"]["num_classes"],
        device=device
    )

    camera = RealSenseCamera(width=640, height=480, fps=30)
    camera.start()

    try:
        while True:
            image_rgb, cloud = camera.get_frame()

            if image_rgb is None or cloud is None:
                continue

            masks, boxes, scores = run_segmentation(
                model=model,
                image_rgb=image_rgb,
                device=device,
                confidence_threshold=config["model"]["confidence_threshold"],
                mask_threshold=config["model"]["mask_threshold"]
            )

            best_obj = None
            grasp = None
            ranked_objects = []

            if len(masks) > 0:
                features = build_features_for_scene(
                    masks=masks,
                    scores=scores,
                    cloud=cloud,
                    z_mode=config["grasp"]["z_mode"]
                )

                ranked_objects = rank_objects(
                    features=features,
                    weights=config["priority"]["weights"]
                )

                if len(ranked_objects) > 0:
                    best_obj = ranked_objects[0]
                    best_mask = masks[best_obj["object_id"]]

                    grasp, candidates, heatmap = find_grasp_point(
                        mask=best_mask,
                        cloud=cloud,
                        weights=config["grasp"]["weights"],
                        window_size=config["grasp"]["window_size"],
                        candidate_stride=config["grasp"]["candidate_stride"],
                        max_candidates=config["grasp"]["max_candidates"],
                        min_valid_ratio=config["grasp"]["min_valid_ratio"],
                        min_edge_distance_px=config["grasp"]["min_edge_distance_px"]
                    )

            frame = draw_live_view(
                image_rgb=image_rgb,
                masks=masks,
                ranked_objects=ranked_objects,
                best_obj=best_obj,
                grasp=grasp
            )

            cv2.imshow("bin picking live", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("error:", e)
        sys.exit(1)