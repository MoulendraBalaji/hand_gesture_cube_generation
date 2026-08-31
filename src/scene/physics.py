"""Physics for released objects: inertia, gravity, floor, and collisions.

When an object is released from a grab it inherits a velocity estimated from
its recent motion.  While marked as ``dynamic`` it is integrated every frame
with gravity, damped against air resistance, and bounces off a virtual floor
plane with restitution.  Loose bounding-sphere collisions between dynamic
objects are also supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RigidBody:
    """A moving scene body with position, velocity, and scale."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 1.0          # collision sphere radius (world units)
    mass: float = 1.0
    dynamic: bool = False        # True => affected by gravity/integration
    grounded: bool = False       # last frame touching the floor


class PhysicsEngine:
    """Semi-implicit Euler integrator with floor + sphere collisions."""

    def __init__(
        self,
        gravity: float = 9.8,
        floor_y: float = -3.0,
        restitution: float = 0.55,
        damping: float = 0.85,
        air_resistance: float = 0.995,
        sphere_collision: bool = True,
    ) -> None:
        self.gravity = gravity
        self.floor_y = floor_y
        self.restitution = restitution
        self.damping = damping
        self.air_resistance = air_resistance
        self.sphere_collision = sphere_collision

    # ------------------------------------------------------------------
    @staticmethod
    def estimate_velocity(
        positions: list[np.ndarray], dt: float
    ) -> np.ndarray:
        """Estimate velocity from the last few positions (finite difference).

        Args:
            positions: Chronological list of 3D positions (most recent last).
            dt: Seconds per step.

        Returns:
            A velocity vector, or zeros if not enough samples.
        """
        if len(positions) < 2:
            return np.zeros(3)
        total = np.zeros(3)
        count = 0
        for i in range(1, len(positions)):
            step = positions[i] - positions[i - 1]
            total += step
            count += 1
        if count == 0 or dt <= 0:
            return np.zeros(3)
        return total / (count * dt)

    # ------------------------------------------------------------------
    def step(
        self,
        body: RigidBody,
        dt: float,
        velocity_override: np.ndarray | None = None,
    ) -> None:
        """Advance a body by one timestep if it is dynamic.

        Args:
            body: The body to integrate.
            dt: Timestep in seconds.
            velocity_override: If given, replaced the body's velocity first
                (used when releasing — inject the estimated launch velocity).
        """
        if not body.dynamic:
            return
        if velocity_override is not None:
            body.velocity = np.asarray(velocity_override, dtype=float).copy()

        # Semi-implicit Euler.
        body.velocity[1] -= self.gravity * dt
        body.velocity *= self.air_resistance
        body.position = body.position + body.velocity * dt

        # Floor collision.
        if body.position[1] < self.floor_y:
            body.position[1] = self.floor_y
            if body.velocity[1] < 0:
                body.velocity[1] = -body.velocity[1] * self.restitution
                body.velocity[0] *= self.damping
                body.velocity[2] *= self.damping
            # Stop tiny vibrations.
            if abs(body.velocity[1]) < 0.05:
                body.velocity[1] = 0.0
            body.grounded = True
        else:
            body.grounded = False

    # ------------------------------------------------------------------
    def collide_spheres(self, bodies: list[RigidBody]) -> None:
        """Resolve loose pairwise sphere collisions between dynamic bodies."""
        if not self.sphere_collision:
            return
        n = len(bodies)
        for i in range(n):
            a = bodies[i]
            if not a.dynamic:
                continue
            for j in range(i + 1, n):
                b = bodies[j]
                r_sum = a.radius + b.radius
                delta = b.position - a.position
                dist_sq = float(delta @ delta)
                if dist_sq > r_sum * r_sum or dist_sq <= 1e-9:
                    continue
                dist = float(np.sqrt(dist_sq))
                overlap = (r_sum - dist) / 2.0
                n_hat = delta / dist
                # Separate on first axis (a dynamic+static handling kept simple).
                a.position -= n_hat * overlap
                b.position += n_hat * overlap

                # Exchange normal velocity components (approx elastic).
                rel = b.velocity - a.velocity
                vn = float(rel @ n_hat)
                if vn < 0:
                    exchange = vn * 0.5
                    a.velocity -= n_hat * exchange
                    b.velocity += n_hat * exchange


class BodyCacher:
    """Tracks recent positions per object to estimate release velocity."""

    def __init__(self, window: int = 8) -> None:
        self.window = window
        self._history: dict[int, list[np.ndarray]] = {}

    def push(self, object_id: int, position: np.ndarray) -> None:
        hist = self._history.setdefault(object_id, [])
        hist.append(np.asarray(position, dtype=float).copy())
        if len(hist) > self.window:
            hist.pop(0)

    def velocity(self, object_id: int, dt: float) -> np.ndarray:
        hist = self._history.get(object_id, [])
        return PhysicsEngine.estimate_velocity(hist, dt)

    def clear(self, object_id: int | None = None) -> None:
        if object_id is None:
            self._history.clear()
        else:
            self._history.pop(object_id, None)
