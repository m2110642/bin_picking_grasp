import numpy as np


def normalize_vector(vector):
    vector = np.array(vector, dtype=float)

    norm = np.linalg.norm(vector)

    if norm < 1e-8:
        return [0.0, 0.0, 0.0]

    return (vector / norm).tolist()


def get_approach_vector_from_normal(normal):
    if normal is None:
        return None

    normal = np.array(normal, dtype=float)
    approach = -normal

    return normalize_vector(approach)


def build_grasp_parameters(best_obj, grasp):
    normal = grasp.get("normal")
    approach_vector = get_approach_vector_from_normal(normal)

    grasp_parameters = {
        "gripper_type": "vacuum",
        "coordinate_system": "camera",

        "object_rank": int(best_obj["rank"]),
        "object_id": int(best_obj["object_id"]),
        "object_priority": float(best_obj["priority"]),

        "point_2d_px": [
            int(grasp["x"]),
            int(grasp["y"])
        ],

        "point_3d_camera_m": [
            float(grasp["point_3d"][0]),
            float(grasp["point_3d"][1]),
            float(grasp["point_3d"][2])
        ],

        "surface_normal_camera": normal,
        "approach_vector_camera": approach_vector,

        "quality": {
            "grasp_score": float(grasp["grasp_score"]),
            "valid_cloud_ratio": float(grasp["valid_cloud_ratio"]),
            "flatness_score": float(grasp["flatness_score"]),
            "edge_distance_score": float(grasp["edge_distance_score"]),
            "normal_score": float(grasp["normal_score_norm"]),
            "centrality_score": float(grasp["centrality_score"])
        }
    }

    return grasp_parameters