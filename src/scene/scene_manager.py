"""Scene model, object lifecycle, selection, and undo/redo.

Defines :class:`SceneObject` (a transformable, physics-capable primitive) and
:class:`SceneManager`, the authoritative store of objects plus an undo/redo
command stack that records every mutating operation (spawn, delete, transform).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .physics import RigidBody
from .primitives import PRIMITIVE_FACTORIES, Mesh


@dataclass
class SceneObject:
    """A placed primitive in the world."""

    obj_id: int
    kind: str                      # "cube" | "pyramid" | "sphere"
    mesh: Mesh
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation_deg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: float = 1.0
    body: RigidBody = field(default_factory=RigidBody)
    color: tuple[int, int, int] = (200, 110, 40)
    selected: bool = False

    @property
    def world_vertices(self) -> np.ndarray:
        """Mesh vertices transformed into world space (position/rotation/scale)."""
        return self._transform(self.mesh.vertices)

    def _transform(self, verts: np.ndarray) -> np.ndarray:
        from . import transform as T

        m = T.compose(
            T.translation(*self.position.tolist()),
            T.rotation_x(self.rotation_deg[0]),
            T.rotation_y(self.rotation_deg[1]),
            T.rotation_z(self.rotation_deg[2]),
            T.uniform_scaling(self.scale),
        )
        return T.apply_matrix(m, verts)

    def clone(self) -> SceneObject:
        """Deep-copy this object (used by undo snapshots)."""
        return SceneObject(
            obj_id=self.obj_id,
            kind=self.kind,
            mesh=self.mesh,
            position=self.position.copy(),
            rotation_deg=self.rotation_deg.copy(),
            scale=self.scale,
            body=RigidBody(
                position=self.body.position.copy(),
                velocity=self.body.velocity.copy(),
                radius=self.body.radius,
                mass=self.body.mass,
                dynamic=self.body.dynamic,
                grounded=self.body.grounded,
            ),
            color=self.color,
            selected=self.selected,
        )


class SceneManager:
    """Owns the object list, selection, next-id counter, and undo/redo stack."""

    def __init__(self) -> None:
        self.objects: list[SceneObject] = []
        self.selected_id: int | None = None
        self._next_id = itertools.count(1)
        self._undo: list[list[SceneObject]] = []
        self._redo: list[list[SceneObject]] = []
        self._max_history = 100

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def selected(self) -> SceneObject | None:
        """Return the currently selected object, if any."""
        if self.selected_id is None:
            return None
        return self.get(self.selected_id)

    def get(self, obj_id: int) -> SceneObject | None:
        for obj in self.objects:
            if obj.obj_id == obj_id:
                return obj
        return None

    def last(self) -> SceneObject | None:
        return self.objects[-1] if self.objects else None

    # ------------------------------------------------------------------
    # Mutations (each records undo state)
    # ------------------------------------------------------------------
    def _snapshot(self) -> list[SceneObject]:
        """Clone every object so undo/redo restores a full prior state."""
        return [o.clone() for o in self.objects]

    def _push_undo(self) -> None:
        self._undo.append(self._snapshot())
        if len(self._undo) > self._max_history:
            self._undo.pop(0)
        self._redo.clear()

    def spawn(
        self,
        kind: str,
        position: np.ndarray,
        rotation_deg: np.ndarray | None = None,
        scale: float = 1.0,
        color: tuple[int, int, int] = (200, 110, 40),
    ) -> SceneObject:
        """Create and add a primitive of *kind* at *position*."""
        factory = PRIMITIVE_FACTORIES.get(kind, PRIMITIVE_FACTORIES["cube"])
        mesh = factory()
        obj = SceneObject(
            obj_id=next(self._next_id),
            kind=kind,
            mesh=mesh,
            position=np.asarray(position, dtype=float),
            rotation_deg=np.asarray(
                rotation_deg if rotation_deg is not None else [0, 0, 0], dtype=float
            ),
            scale=scale,
            body=RigidBody(position=np.asarray(position, dtype=float).copy()),
            color=color,
        )
        self._push_undo()
        self.objects.append(obj)
        self.select(obj.obj_id)
        return obj

    def delete(self, obj_id: int) -> None:
        obj = self.get(obj_id)
        if obj is None:
            return
        self._push_undo()
        self.objects = [o for o in self.objects if o.obj_id != obj_id]
        if self.selected_id == obj_id:
            self.selected_id = None

    def delete_selected(self) -> None:
        if self.selected_id is not None:
            self.delete(self.selected_id)

    def select(self, obj_id: int | None) -> None:
        if self.selected_id is not None and self.get(self.selected_id):
            self.get(self.selected_id).selected = False
        self.selected_id = obj_id
        if obj_id is not None and self.get(obj_id):
            self.get(obj_id).selected = True

    # ------------------------------------------------------------------
    # Transform operations (each records undo state)
    # ------------------------------------------------------------------
    def move(self, obj_id: int, delta: np.ndarray) -> None:
        obj = self.get(obj_id)
        if obj is None:
            return
        self._push_undo()
        obj.position = obj.position + np.asarray(delta, dtype=float)

    def set_position(self, obj_id: int, position: np.ndarray) -> None:
        obj = self.get(obj_id)
        if obj is None:
            return
        self._push_undo()
        obj.position = np.asarray(position, dtype=float)

    def rotate(self, obj_id: int, axis: str, deg: float) -> None:
        obj = self.get(obj_id)
        if obj is None:
            return
        self._push_undo()
        axes = {"x": 0, "y": 1, "z": 2}
        obj.rotation_deg[axes[axis]] += deg

    def scale_by(self, obj_id: int, factor: float) -> None:
        obj = self.get(obj_id)
        if obj is None:
            return
        self._push_undo()
        obj.scale = max(0.1, min(8.0, obj.scale * factor))

    # ------------------------------------------------------------------
    # Undo / redo
    # ------------------------------------------------------------------
    def undo(self) -> bool:
        """Restore the previous scene state.

        Returns:
            ``True`` if an undo happened.
        """
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self.objects = self._undo.pop()
        if self.selected_id is not None and self.get(self.selected_id) is None:
            self.selected_id = None
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self.objects = self._redo.pop()
        if self.selected_id is not None and self.get(self.selected_id) is None:
            self.selected_id = None
        return True

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def clear(self) -> None:
        if not self.objects:
            return
        self._push_undo()
        self.objects.clear()
        self.selected_id = None
