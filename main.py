import os
import sys
from pathlib import Path

import numpy as np
import torch

from src.model_loader import load_mask_rcnn_model
from src.segmenter import read_rgb_image, run_segmentation
from src.object_prioritizer import build_features_for_scene, rank_objects
from src.grasp_detector import find_grasp_point
from src.visualizer import draw_pipeline_result
from src.grasp_parameters import build_grasp_parameters
from src.utils import (
    load_config,
    resolve_path,
    ensure_dir,
    save_json,
    load_cloud,
    find_rgb_cloud_pairs,
)


def main():
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Не найден config.yaml: {config_path}")

    config = load_config(str(config_path))

    images_dir = resolve_path(str(project_root), config["paths"]["images_dir"])
    cloud_dir = resolve_path(str(project_root), config["paths"]["cloud_dir"])
    model_path = resolve_path(str(project_root), config["paths"]["model_path"])
    output_dir = resolve_path(str(project_root), config["paths"]["output_dir"])

    ensure_dir(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("Bin Picking Grasp Pipeline")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Device: {device}")
    print(f"Images dir: {images_dir}")
    print(f"Cloud dir: {cloud_dir}")
    print(f"Model path: {model_path}")
    print(f"Output dir: {output_dir}")
    print("=" * 70)

    model = load_mask_rcnn_model(
        model_path=model_path,
        num_classes=config["model"]["num_classes"],
        device=device
    )

    pairs = find_rgb_cloud_pairs(images_dir, cloud_dir)

    if len(pairs) == 0:
        raise RuntimeError(
            "Не найдено ни одной пары RGB + cloud. "
            "Проверьте папки data/images и data/cloud."
        )


    all_results = []

    for scene_index, pair in enumerate(pairs, start=1):
        print()
        print("-" * 70)
        print(f"[{scene_index}/{len(pairs)}] Обработка сцены: {pair['image_file']}")

        image_rgb = read_rgb_image(pair["image_path"])
        cloud = load_cloud(pair["cloud_path"])

        if cloud.shape[0] != image_rgb.shape[0] or cloud.shape[1] != image_rgb.shape[1]:
            print(
                f"Пропуск: размер cloud {cloud.shape} не совпадает "
                f"с размером RGB {image_rgb.shape}"
            )
            continue

        masks, boxes, scores = run_segmentation(
            model=model,
            image_rgb=image_rgb,
            device=device,
            confidence_threshold=config["model"]["confidence_threshold"],
            mask_threshold=config["model"]["mask_threshold"]
        )

        if len(masks) == 0:
            print("Объекты не найдены.")
            continue

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

        if grasp is None:
            print("Точка захвата не найдена.")
            continue

        grasp_parameters = build_grasp_parameters(
            best_obj=best_obj,
            grasp=grasp
        )

        scene_output_dir = os.path.join(
            output_dir,
            f"scene_{scene_index:02d}_{pair['base_name']}"
        )
        ensure_dir(scene_output_dir)

        visualization_path = os.path.join(scene_output_dir, "grasp_result.png")
        result_json_path = os.path.join(scene_output_dir, "grasp_result.json")
        top_candidates_path = os.path.join(scene_output_dir, "top_grasp_candidates.json")

        if config["visualization"]["save_images"]:
            draw_pipeline_result(
                image_rgb=image_rgb,
                masks=masks,
                ranked_objects=ranked_objects,
                best_obj=best_obj,
                grasp=grasp,
                heatmap=heatmap,
                cloud=cloud,
                save_path=visualization_path,
                crop_padding=config["visualization"]["crop_padding"]
            )

        result = {
            "scene_index": scene_index,
            "image": pair["image_file"],
            "cloud": pair["cloud_file"],
            "num_detected_objects": len(masks),
            "selected_object": best_obj,
            "grasp_point": grasp,
            "grasp_parameters": grasp_parameters,
            "num_grasp_candidates": len(candidates),
            "visualization": visualization_path
        }

        save_json(result, result_json_path)
        save_json(candidates[:25], top_candidates_path)

        summary_item = {
            "scene_index": scene_index,
            "image": pair["image_file"],
            "num_detected_objects": len(masks),
            "selected_object_id": best_obj["object_id"],
            "selected_object_priority": best_obj["priority"],
            "grasp_x": grasp["x"],
            "grasp_y": grasp["y"],
            "grasp_3d": grasp["point_3d"],
            "grasp_score": grasp["grasp_score"],
            "valid_cloud_ratio": grasp["valid_cloud_ratio"],
            "flatness_score": grasp["flatness_score"],
            "edge_distance_score": grasp["edge_distance_score"],
            "normal_score": grasp["normal_score_norm"],
            "num_grasp_candidates": len(candidates),
            "grasp_parameters": grasp_parameters,
            "visualization": visualization_path
        }

        all_results.append(summary_item)

        print(f"Найдено объектов: {len(masks)}")
        print(f"Выбран объект ID: {best_obj['object_id']}")
        print(f"Priority score: {best_obj['priority']:.4f}")
        print(f"Точка захвата: x={grasp['x']}, y={grasp['y']}")
        print(f"Grasp score: {grasp['grasp_score']:.4f}")
        print(f"Кандидатов рассмотрено: {len(candidates)}")
        print(f"Результат сохранён: {scene_output_dir}")

    summary_path = os.path.join(output_dir, "pipeline_summary.json")
    save_json(all_results, summary_path)

    print()
    print("Готово")
    print(f"Обработано сцен: {len(all_results)}")
    print(f"Файл: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("Ошибка:")
        print(error)
        sys.exit(1)