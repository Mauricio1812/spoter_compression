import math
import logging
import cv2
import random
import numpy as np

def __rotate(origin: tuple, point: tuple, angle: float):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def augment_rotate(sign: np.ndarray, angle_range: tuple) -> np.ndarray:
    angle = math.radians(random.uniform(*angle_range))
    origin = (0.5, 0.5)
    num_frames, num_points, _ = sign.shape

    rotated_sign = np.copy(sign)
    for frame_idx in range(num_frames):
        for point_idx in range(num_points):
            x, y = sign[frame_idx, point_idx, :]
            rotated_sign[frame_idx, point_idx, :] = __rotate(origin, (x, y), angle)
    return rotated_sign


def augment_shear(sign: np.ndarray, type: str, squeeze_ratio: tuple) -> np.ndarray:
    num_frames, num_points, _ = sign.shape

    if type == "squeeze":
        move_left = random.uniform(*squeeze_ratio)
        move_right = random.uniform(*squeeze_ratio)
        src = np.array([[0, 1], [1, 1], [0, 0], [1, 0]], dtype=np.float32)
        dest = np.array([[0 + move_left, 1], [1 - move_right, 1], [0 + move_left, 0], [1 - move_right, 0]], dtype=np.float32)
        mtx = cv2.getPerspectiveTransform(src, dest)
    elif type == "perspective":
        move_ratio = random.uniform(*squeeze_ratio)
        src = np.array([[0, 1], [1, 1], [0, 0], [1, 0]], dtype=np.float32)
        if random.random() < 0.5:
            dest = np.array([[0 + move_ratio, 1 - move_ratio], [1, 1], [0 + move_ratio, 0 + move_ratio], [1, 0]], dtype=np.float32)
        else:
            dest = np.array([[0, 1], [1 - move_ratio, 1 - move_ratio], [0, 0], [1 - move_ratio, 0 + move_ratio]], dtype=np.float32)
        mtx = cv2.getPerspectiveTransform(src, dest)
    else:
        logging.error("Unsupported shear type provided.")
        return sign

    augmented_sign = np.copy(sign)
    for frame_idx in range(num_frames):
        frame = sign[frame_idx]
        augmented_frame = cv2.perspectiveTransform(np.array([frame], dtype=np.float32), mtx)[0]
        augmented_sign[frame_idx] = augmented_frame
    return augmented_sign
