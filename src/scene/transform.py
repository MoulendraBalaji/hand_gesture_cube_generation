"""3D linear algebra for rendering and object transforms.

Implements the homogeneous-coordinate machinery used everywhere else: model
transforms (translation / rotation / non-uniform scaling), quaternion rotation,
and a full view + perspective projection pipeline.  Pure ``numpy`` with no
graphics dependency, so it is fully unit-testable headlessly.

Conventions:
    * Column-vector layout: a point ``p`` is a shape-(3,) or (N,3) ``ndarray``.
    * Points are transformed right-to-left, ``p' = M @ p``.
"""

from __future__ import annotations

import math

import numpy as np

Vec = np.ndarray


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------
def translation(tx: float, ty: float, tz: float) -> np.ndarray:
    """4x4 translation matrix."""
    t = np.eye(4)
    t[0, 3] = tx
    t[1, 3] = ty
    t[2, 3] = tz
    return t


def scaling(sx: float, sy: float, sz: float) -> np.ndarray:
    """4x4 non-uniform scale matrix."""
    s = np.eye(4)
    s[0, 0] = sx
    s[1, 1] = sy
    s[2, 2] = sz
    return s


def uniform_scaling(s: float) -> np.ndarray:
    """4x4 uniform scale matrix."""
    d = np.eye(4) * s
    d[3, 3] = 1.0
    return d


def rotation_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[1, 1] = c
    m[1, 2] = -s
    m[2, 1] = s
    m[2, 2] = c
    return m


def rotation_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[0, 0] = c
    m[0, 2] = s
    m[2, 0] = -s
    m[2, 2] = c
    return m


def rotation_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[0, 0] = c
    m[0, 1] = -s
    m[1, 0] = s
    m[1, 1] = c
    return m


def look_at(eye: Vec, target: Vec, up: Vec) -> np.ndarray:
    """Build a right-handed view matrix looking from *eye* at *target*."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    m = np.eye(4)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def perspective(
    fov_deg: float, aspect: float, near: float, far: float
) -> np.ndarray:
    """Right-handed perspective projection matrix."""
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


# ---------------------------------------------------------------------------
# Quaternions
# ---------------------------------------------------------------------------
def quat_from_axis_angle(axis: Vec, deg: float) -> np.ndarray:
    """Return a unit quaternion ``[w, x, y, z]`` from an axis + angle."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / norm
    half = math.radians(deg) / 2.0
    s = math.sin(half)
    return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions ``[w, x, y, z]``."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion ``[w, x, y, z]`` to a 4x4 rotation matrix."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(4)
    s = 2.0 / n
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz = y * ys, y * zs
    zz = z * zs
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy, 0.0],
        [xy + wz, 1.0 - (xx + zz), yz - wx, 0.0],
        [xz - wy, yz + wx, 1.0 - (xx + yy), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def rotate_quat(q: np.ndarray, v: Vec) -> Vec:
    """Rotate a 3-vector (or 3 rows) by a unit quaternion."""
    m = quat_to_matrix(q)[:3, :3]
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        return m @ v
    return (m @ v.T).T


# ---------------------------------------------------------------------------
# Transform helper
# ---------------------------------------------------------------------------
def compose(*matrices: np.ndarray) -> np.ndarray:
    """Multiply a sequence of 4x4 matrices (leftmost applied last)."""
    m = np.eye(4)
    for mat in matrices:
        m = mat @ m
    return m


def apply_matrix(m: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Transform an (N,3) array of points by a 4x4 matrix (perspective divide)."""
    pts = np.asarray(points, dtype=float)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    out = (m @ pts_h.T).T
    w = out[:, 3:4]
    w[w == 0] = 1e-12
    return out[:, :3] / w


def project_to_screen(points_3d, width: int, height: int) -> np.ndarray:
    """Map NDC-ish camera-space points to pixel coordinates (sanity helper)."""
    pts = np.asarray(points_3d, dtype=float)
    x = (pts[:, 0] + 1.0) * 0.5 * width
    y = (1.0 - (pts[:, 1] + 1.0) * 0.5) * height
    return np.column_stack([x, y])
