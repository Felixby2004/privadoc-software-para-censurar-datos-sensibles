from __future__ import annotations

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype="float32").reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    out = np.zeros((4, 2), dtype="float32")
    out[0] = pts[np.argmin(s)]      # top-left
    out[2] = pts[np.argmax(s)]      # bottom-right
    out[1] = pts[np.argmin(d)]      # top-right
    out[3] = pts[np.argmax(d)]      # bottom-left
    return out


def perspective_matrix(quad: np.ndarray, width: int, height: int) -> np.ndarray:
    src = order_points(quad)
    # Ensure landscape mapping even if the contour arrived portrait-oriented.
    ew = max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3]))
    eh = max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1]))
    if eh > ew:
        src = np.array([src[3], src[0], src[1], src[2]], dtype="float32")
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    return cv2.getPerspectiveTransform(src, dst)


def warp_with_matrix(img: np.ndarray, H: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.warpPerspective(img, H, (width, height), flags=cv2.INTER_CUBIC)


def rotation_homography(width: int, height: int, angle: int) -> tuple[np.ndarray, tuple[int, int]]:
    angle %= 360
    if angle == 0:
        return np.eye(3, dtype=np.float64), (width, height)
    if angle == 90:
        # (x,y) -> (h-1-y, x)
        H = np.array([[0, -1, height - 1], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        return H, (height, width)
    if angle == 180:
        H = np.array([[-1, 0, width - 1], [0, -1, height - 1], [0, 0, 1]], dtype=np.float64)
        return H, (width, height)
    if angle == 270:
        # (x,y) -> (y, w-1-x)
        H = np.array([[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]], dtype=np.float64)
        return H, (height, width)
    raise ValueError("Ángulo no soportado; usa 0, 90, 180 o 270.")


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    angle %= 360
    if angle == 0:
        return img.copy()
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("Ángulo no soportado.")


def transform_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def box_to_polygon(box: list[int], padding: tuple[int, int] = (0, 0)) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in box]
    px, py = padding
    x1 -= px; x2 += px; y1 -= py; y2 += py
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
