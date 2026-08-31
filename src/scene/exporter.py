"""Export the current scene to a valid Wavefront ``.obj`` file.

Writes vertex / face data for every object (transformed into world space) with
a per-object ``o`` group so the result opens cleanly in Blender, MeshLab, or
any standard OBJ viewer.  Each object's material is approximated with a ``usemtl``
line and a simple ``mtllib`` is emitted for colour fidelity.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..utils.logger import get_logger
from .primitives import Mesh
from .scene_manager import SceneObject

log = get_logger("scene.exporter")


def mesh_to_obj(meshes: Sequence[tuple[str, Mesh, tuple]], obj_path: str | Path) -> Path:
    """Write a set of named meshes to an OBJ file.

    Args:
        meshes: Sequence of ``(name, mesh, color_bgr)`` tuples.
        obj_path: Destination ``.obj`` file path.

    Returns:
        The path written.
    """
    out = Path(obj_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# GestureForge scene export",
        "# Generated automatically by gestureforge.scene.exporter",
        "",
        "mtllib scene.mtl",
    ]
    vert_offset = 0
    for name, mesh, color_bgr in meshes:
        lines.append(f"\no {name}")
        for v in mesh.vertices:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        r, g, b = (c / 255.0 for c in reversed(color_bgr))  # BGR -> RGB
        lines.append(f"usemtl mat_{name}")
        for face in mesh.faces:
            indices = " ".join(str(i + 1 + vert_offset) for i in face)
            lines.append(f"f {indices}")
        vert_offset += mesh.vertex_count

    # Emit a companion material file so colours survive in Blender.
    mtl_path = out.with_suffix(".mtl")
    mtl_lines = ["# GestureForge material library", ""]
    for name, _mesh, color_bgr in meshes:
        r, g, b = (c / 255.0 for c in reversed(color_bgr))
        mtl_lines.extend(
            [
                f"newmtl mat_{name}",
                f"Kd {r:.3f} {g:.3f} {b:.3f}",
                "Ks 0.0 0.0 0.0",
                "d 1.0",
                "illum 1",
                "",
            ]
        )
    mtl_path.write_text("\n".join(mtl_lines), encoding="utf-8")

    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("Exported %d objects to %s", len(meshes), out)
    return out


def export_scene(objects: Sequence[SceneObject], obj_path: str | Path) -> Path:
    """Export :class:`SceneObject`\\ s (with colors) to an OBJ file."""
    meshes = [
        (f"obj_{o.obj_id}_{o.kind}", _to_world_mesh(o), o.color)
        for o in objects
    ]
    return mesh_to_obj(meshes, obj_path)


def _to_world_mesh(obj: SceneObject) -> Mesh:
    """Return a mesh whose vertices are already in world space."""
    from . import transform as T

    m = T.compose(
        T.translation(*obj.position.tolist()),
        T.rotation_x(obj.rotation_deg[0]),
        T.rotation_y(obj.rotation_deg[1]),
        T.rotation_z(obj.rotation_deg[2]),
        T.uniform_scaling(obj.scale),
    )
    world_verts = T.apply_matrix(m, obj.mesh.vertices)
    return Mesh(world_verts, obj.mesh.faces)
