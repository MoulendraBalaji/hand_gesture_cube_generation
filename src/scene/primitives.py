"""Parametric 3D primitive meshes.

Generators for Cube, Pyramid, and wireframe-Sphere as explicit vertex/face
lists so they can be projected by the OpenCV renderer, shaded by the GL
renderer, and exported to ``.obj`` from one shared definition.  Vertices are
produced centered on the origin with unit-ish size; the scene manager applies
position / rotation / scale.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class Mesh:
    """A simple renderable mesh: vertices + indexed faces."""

    def __init__(self, vertices: np.ndarray, faces: list[list[int]] | np.ndarray) -> None:
        self.vertices: np.ndarray = np.asarray(vertices, dtype=float)  # (N,3) or (N,4)
        self.faces: list[list[int]] = [
            [int(i) for i in face] for face in faces
        ]

    @property
    def vertex_count(self) -> int:
        return self.vertices.shape[0]

    @property
    def face_count(self) -> int:
        return len(self.faces)


def cube_mesh(size: float = 2.0) -> Mesh:
    """Axis-aligned wireframe cube centered at the origin, half-extent *size*."""
    h = size / 2.0
    verts = np.array([
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
        [-h, -h,  h], [h, -h,  h], [h, h,  h], [-h, h,  h],
    ], dtype=float)
    faces = [
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5],
    ]
    return Mesh(verts, faces)


def pyramid_mesh(size: float = 2.0, apex_offset: float = 1.2) -> Mesh:
    """Square-based pyramid centered so its base sits near the origin."""
    h = size / 2.0
    base_z = -apex_offset * h  # keep apex at +apex
    verts = np.array([
        [-h, -h, base_z], [h, -h, base_z], [h, h, base_z], [-h, h, base_z],  # base
        [0.0, 0.0, h * apex_offset],                                          # apex
    ], dtype=float)
    faces = [
        [0, 1, 2, 3],   # base
        [0, 1, 4],      # front
        [1, 2, 4],      # right
        [2, 3, 4],      # back
        [3, 0, 4],      # left
    ]
    return Mesh(verts, faces)


def sphere_mesh(radius: float = 1.0, stacks: int = 8, slices: int = 12) -> Mesh:
    """Wireframe sphere built from latitude/longitude quads."""
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    for i in range(stacks + 1):
        phi = math.pi * i / stacks            # 0..pi
        for j in range(slices):
            theta = 2.0 * math.pi * j / slices
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.cos(phi)
            z = radius * math.sin(phi) * math.sin(theta)
            verts.append([x, y, z])
    def idx(i: int, j: int) -> int:
        return i * slices + (j % slices)
    for i in range(stacks):
        for j in range(slices):
            a = idx(i, j)
            b = idx(i + 1, j)
            c = idx(i + 1, j + 1)
            d = idx(i, j + 1)
            faces.append([a, b, c, d])
    return Mesh(np.array(verts, dtype=float), faces)


def edge_connectivity(mesh: Mesh) -> list[tuple[int, int]]:
    """Derive unique undirected edges from a mesh's faces (for wireframe)."""
    edges: set[tuple[int, int]] = set()
    for face in mesh.faces:
        n = len(face)
        for k in range(n):
            a, b = face[k], face[(k + 1) % n]
            edges.add((min(a, b), max(a, b)))
    return sorted(edges)


def north_pole_normal(vertex: np.ndarray) -> np.ndarray:
    """Approximate surface normal for Lambert shading from the mesh origin."""
    return vertex / (np.linalg.norm(vertex) + 1e-12)


PRIMITIVE_FACTORIES: dict[str, Any] = {
    "cube": cube_mesh,
    "pyramid": pyramid_mesh,
    "sphere": sphere_mesh,
}

PRIMITIVE_ORDER: tuple[str, ...] = ("cube", "pyramid", "sphere")
