"""Core application loop / state machine for GestureForge.

Wires the vision, gesture, scene, render, physics, UI, analytics, voice, and
macro subsystems into a single real-time loop.  The loop:

    1. capture a frame from the webcam,
    2. run hand detection (smoothly, off-thread via the pipeline),
    3. detect gestures (rule-based + optional ML vote),
    4. apply raw keyboard and gesture actions,
    5. step physics, render the scene, draw the HUD, and present.

Every action has a keyboard fallback so the app is demo-able even when a
gesture misfires on camera.  Missing optional subsystems (voice, OpenGL, ML)
degrade gracefully to a reduced mode rather than crashing.
"""

from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path

import numpy as np

from .analytics.report_generator import generate_report
from .analytics.session_logger import SessionLogger
from .gestures import GestureEvent
from .gestures.macro import MacroPlayer, MacroRecorder
from .gestures.ml_classifier import GestureClassifier
from .gestures.rule_based import RuleBasedGestureRecognizer
from .gestures.trainer import GestureTrainer
from .render.base_renderer import BaseRenderer
from .render.cv_renderer import OpenCVRenderer
from .scene.exporter import export_scene
from .scene.physics import BodyCacher, PhysicsEngine
from .scene.primitives import PRIMITIVE_ORDER
from .scene.scene_manager import SceneManager
from .ui.hud import HUD
from .utils.config_loader import Config
from .utils.logger import get_logger
from .vision.depth_estimator import DepthEstimator
from .vision.hand_tracker import LM, Hand, HandTracker, HandTrackerPipeline

log = get_logger("app")

MODE_NORMAL = "normal"
MODE_DRAW = "draw"
MODE_TRAIN = "train"


class GestureForgeApp:
    """The top-level application state machine."""

    def __init__(self, config: Config, *, headless_log_only: bool = False) -> None:
        self.config = config
        self.headless_log_only = headless_log_only

        # Subsystems (created on demand to keep construction cheap for tests).
        self.tracker: HandTrackerPipeline | None = None
        self.recognizer = RuleBasedGestureRecognizer(
            pinch_ratio=config.get("calibration.pinch_ratio", 0.35),
            fist_ratio=config.get("calibration.fist_ratio", 0.45),
        )
        self.classifier = GestureClassifier()
        self.trainer = GestureTrainer(classifier=self.classifier)
        self.scene = SceneManager()
        self.physics = PhysicsEngine(
            gravity=config.get("physics.gravity", 9.8),
            floor_y=config.get("physics.floor", -3.0),
            restitution=config.get("physics.restitution", 0.55),
            damping=config.get("physics.damping", 0.85),
            air_resistance=config.get("physics.air_resistance", 0.995),
            sphere_collision=config.get("physics.sphere_collision", True),
        )
        self.body_cacher = BodyCacher()
        self.depth = DepthEstimator(
            hand_span_m=config.get("calibration.hand_span_m", 0.19),
            reference_distance_m=config.get("calibration.reference_distance", 0.6),
        )
        self.session = SessionLogger(
            db_path=config.get("analytics.database", "output/session.db"),
            csv_path=config.get("analytics.csv_path", "output/session.csv"),
        )
        self.hud = HUD(title=config.get("app.name", "GestureForge"))
        self.renderer: BaseRenderer | None = None

        self.macro_recorder = MacroRecorder()
        self.macro_player = MacroPlayer(self.dispatch_action)

        # State
        self.mode = MODE_NORMAL
        self.calibrated = False
        self.last_gesture = "none"
        self.last_confidence = 0.0
        self.fps = 0.0
        self._frame_dims: tuple[int, int] = (0, 0)
        self._palm_spans: deque[float] = deque(maxlen=5)
        self._grab_held = False
        self._grabbed_id: int | None = None
        self._grabbed_history: dict[int, deque] = {}
        self._calibrate_until = 0.0
        self._arm_ok_t0: float | None = None
        self._spawn_lock_until = 0.0
        self._draw_canvas: np.ndarray | None = None
        self._last_draw_pt: tuple[int, int] | None = None
        self._motion_hist: dict[int, deque] = {}
        self._prev_twist_angle: float | None = None
        self._prev_two_hand_span: float | None = None
        self._current_kind = PRIMITIVE_ORDER[0]
        self._frame_index = 0

        # Voice (optional)
        self.voice = None
        self._renderer_name = config.get("renderer.default", "opencv")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def run(self, camera_index: int, renderer_name: str | None = None) -> None:
        """Start the camera loop and run until quit."""
        self._renderer_name = renderer_name or self._renderer_name
        self.session.open()
        if self.config.get("gestures.use_ml", True):
            self.trainer.load_if_available()

        device_ok = self._setup_device(camera_index)
        if not device_ok and self.headless_log_only:
            log.info("Headless log-only mode: no camera attached, exiting loop.")
            self.session.close()
            return

        self._setup_renderer()
        if not device_ok:
            log.error("No webcam available; cannot start the interaction loop.")
            self._shutdown_cleanup()
            return

        if self.config.get("app.mirror", True):
            self._flip = True
        else:
            self._flip = False

        if self.config.get("voice.enabled", False):
            self._start_voice()

        log.info("GestureForge starting on camera %d", camera_index)
        self._loop()

    def _setup_device(self, camera_index: int) -> bool:
        import cv2

        cam_cfg = self.config.section("camera")
        width = cam_cfg.get("width", 1280)
        height = cam_cfg.get("height", 720)
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return False
        self._cap = cap
        self._frame_dims = (frame.shape[1], frame.shape[0])
        tracker = HandTracker(
            max_num_hands=self.config.get("vision.max_num_hands", 2),
            min_detection_confidence=self.config.get("vision.min_detection_confidence", 0.5),
            min_tracking_confidence=self.config.get("vision.min_tracking_confidence", 0.5),
            filter_min_cutoff=self.config.get("vision.filter.min_cutoff", 1.0),
            filter_beta=self.config.get("vision.filter.beta", 0.007),
            filter_d_cutoff=self.config.get("vision.filter.d_cutoff", 1.0),
        )
        self.tracker = HandTrackerPipeline(tracker)
        self.tracker.start()
        if self.config.get("calibration.auto_on_first_run", True):
            self._calibrate_until = time.time() + 3.0
        return True

    def _setup_renderer(self) -> None:
        name = self._renderer_name
        if name == "opengl":
            try:
                from .render.gl_renderer import GLRenderer

                self.renderer = GLRenderer(
                    width=self._frame_dims[0] or 1280,
                    height=self._frame_dims[1] or 720,
                    title=self.config.get("app.window_title", "GestureForge"),
                    fov_deg=self.config.get("renderer.cv.perspective_fov_deg", 60),
                    camera_z=self.config.get("renderer.cv.camera_z", 6.0),
                    clear_color=self.config.get("renderer.gl.clear_color", [0.12, 0.12, 0.14, 1.0]),
                )
                log.info("Using OpenGL renderer")
            except Exception as exc:
                log.warning("OpenGL renderer unavailable (%s); falling back to OpenCV", exc)
                self.renderer = None
        if self.renderer is None:
            cv_cfg = self.config.section("renderer").section("cv")
            self.renderer = OpenCVRenderer(
                fov_deg=cv_cfg.get("perspective_fov_deg", 60),
                near=cv_cfg.get("near", 0.1),
                far=cv_cfg.get("far", 100.0),
                camera_z=cv_cfg.get("camera_z", 6.0),
                mesh_color=tuple(cv_cfg.get("mesh_color", [200, 110, 40])),
                edge_color=tuple(cv_cfg.get("edge_color", [255, 235, 200])),
                light_dir=tuple(cv_cfg.get("light_dir", [0.4, 0.8, 0.6])),
            )
            log.info("Using OpenCV renderer")

    def _start_voice(self) -> None:
        try:
            from .voice.voice_commands import VoiceListener

            listener = VoiceListener(
                model_dir=self.config.get("voice.model_dir", "assets/voice/vosk-model-small-en-us"),
                sample_rate=self.config.get("voice.sample_rate", 16000),
            )
            if listener.attempt_connect():
                self.voice = listener
                log.info("Voice listener active")
            else:
                log.warning("Voice unavailable: %s", listener.error)
        except Exception as exc:  # pragma: no cover
            log.warning("Voice init failed: %s", exc)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        import cv2

        frame_time = time.perf_counter()
        fps_count = 0
        fps_t0 = time.time()
        target_dt = 1.0 / max(1, self.config.get("app.fps_target", 30))

        try:
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    log.warning("Camera read failed")
                    break
                if self._flip:
                    frame = cv2.flip(frame, 1)

                now = time.perf_counter()
                dt = max(0.0001, now - frame_time)
                frame_time = now

                # Hand detection on background thread (smooth feed).
                if self.tracker is not None:
                    self.tracker.submit(frame.copy())
                    hands = self.tracker.get()
                else:
                    hands = []

                # Gesture detection + ML vote.
                events = self.recognizer.detect(hands)
                gestures = self._merge_ml(hands, events)
                gesture = gestures[0] if gestures else GestureEvent()

                # Voice + keyboard + gesture actions.
                self._drain_voice()
                key = cv2.waitKey(1) & 0xFF
                self._handle_key(key)
                self._apply_gesture(hands, gesture, now)

                # Physics + scene integration.
                self._step_physics(dt)

                # Compose output frame.
                self._draw_hand_overlay(frame, hands)
                out = self._render_scene_frame(frame)
                self._draw_canvas_overlay(out)
                self.hud.update_fps(self.fps)
                cal_state = self._calibration_state()
                self.hud.draw(
                    out,
                    active_gesture=gesture.name if gesture.name != "none" else "—",
                    confidence=gesture.confidence,
                    calibration_state=cal_state,
                    mode=self.mode,
                    scene_info=self._scene_info(),
                )

                self._present(out)

                # FPS accounting.
                fps_count += 1
                if time.time() - fps_t0 >= 0.5:
                    self.fps = fps_count / (time.time() - fps_t0)
                    self.hud.update_fps(self.fps)
                    fps_count = 0
                    fps_t0 = time.time()

                # Throttle to target frame rate.
                elapsed = time.perf_counter() - now
                sleep = target_dt - elapsed
                if sleep > 0:
                    time.sleep(sleep)

                esc = 27
                if key == ord(self.config.get("keys.quit", "q")) or key == esc:
                    self._session_summary()
                    break
                self._frame_index += 1
        finally:
            self._shutdown_cleanup()

    # ------------------------------------------------------------------
    # Gesture ML merge
    # ------------------------------------------------------------------
    def _merge_ml(self, hands: list[Hand], rule_events: list[GestureEvent]) -> list[GestureEvent]:
        if not self.config.get("gestures.use_ml", True) or not self.classifier.is_trained:
            return rule_events
        # ML vote over the first hand; append if confident & different.
        if hands:
            ml_ev = self.classifier.predict(hands[0])
            if ml_ev is not None:
                # Prefer ML result but keep rule events as extra context.
                base = [e for e in rule_events if e.name != ml_ev.name]
                base.insert(0, ml_ev)
                return base
        return rule_events

    # ------------------------------------------------------------------
    # Gesture + key action dispatch
    # ------------------------------------------------------------------
    def dispatch_action(self, action: str, payload: dict = None) -> None:
        """Dispatch a canonical action name (used by macros, voice, gestures)."""
        payload = payload or {}
        if action == "spawn":
            kind = payload.get("kind", PRIMITIVE_ORDER[0])
            self.action_spawn(kind)
        elif action == "spawn_cube":
            self.action_spawn("cube")
        elif action == "spawn_pyramid":
            self.action_spawn("pyramid")
        elif action == "spawn_sphere":
            self.action_spawn("sphere")
        elif action == "delete":
            self.action_delete()
        elif action == "undo":
            self.scene.undo()
            self._log("undo", "undo")
        elif action == "redo":
            self.scene.redo()
            self._log("redo", "redo")
        elif action == "clear_scene":
            self.scene.clear()
            self._log("clear_scene", "clear")
        elif action == "save_report" or action == "save_scene":
            self._save_scene_and_report()
        elif action == "calibrate":
            self._start_calibration()

    def action_spawn(self, kind: str | None = None) -> None:
        if time.time() < self._spawn_lock_until:
            return
        kind = kind or self._current_cycle_kind()
        pos = np.array([0.0, 1.0, 0.0])
        obj = self.scene.spawn(kind, pos, scale=1.2)
        obj.body.dynamic = False
        self.macro_recorder.record("spawn", kind=kind, obj_id=obj.obj_id)
        self._log("spawn", kind, obj.obj_id)
        self._spawn_lock_until = time.time() + 0.4

    def action_delete(self) -> None:
        if self._grabbed_id is not None:
            self.scene.delete(self._grabbed_id)
            self.macro_recorder.record("delete", obj_id=self._grabbed_id)
            self._log("delete", "grabbed", self._grabbed_id)
            self._grabbed_id = None
        else:
            self.scene.delete_selected()
            self._log("delete", "selected")

    def _current_cycle_kind(self) -> str:
        idx = 0
        if self.scene.last() is not None:
            try:
                idx = PRIMITIVE_ORDER.index(self.scene.last().kind)
            except ValueError:
                idx = 0
        return PRIMITIVE_ORDER[(idx + 1) % len(PRIMITIVE_ORDER)]

    def cycle_primitive(self) -> None:
        """Advance the spawn-primitive cycle and persist the choice."""
        self._current_kind = self._current_cycle_kind()
        self._log("cycle", self._current_kind)

    # ------------------------------------------------------------------
    def _handle_key(self, key: int) -> None:
        keys = self.config.section("keys")
        k = chr(key) if 32 <= key < 127 else ""

        # Quit handled by caller.
        if k == keys.get("spawn", " "):
            self.dispatch_action("spawn")
        elif k.lower() == keys.get("grab", "g"):
            self._toggle_grab()
        elif k.lower() == keys.get("air_draw_toggle", "d"):
            self._toggle_draw_mode()
        elif k.lower() == keys.get("calibrate", "c"):
            self.dispatch_action("calibrate")
        elif k in ("+", "="):
            self._scale_selected(1.15)
        elif k in ("-", "_"):
            self._scale_selected(1 / 1.15)
        elif k == keys.get("rotate_y_plus", "]"):
            self._rotate_selected(15.0)
        elif k == keys.get("rotate_y_minus", "["):
            self._rotate_selected(-15.0)
        elif k.lower() == keys.get("toggle_help", "h"):
            self.hud.show_help = not self.hud.show_help
        elif k.lower() == keys.get("toggle_debug", "f"):
            self.hud.show_debug = not self.hud.show_debug
        elif k.lower() == keys.get("cycle_primitive", "tab") or str(key) == "9":
            self.cycle_primitive()
        elif k.lower() == keys.get("save_report", "s"):
            self.action_save_report()
        elif k.lower() == keys.get("record_macro", "r"):
            self._toggle_macro_record()
        elif k.lower() == keys.get("replay_macro", "p"):
            self._toggle_macro_replay()
        elif k.lower() == keys.get("clear_scene", "x"):
            self.dispatch_action("clear_scene")
        elif k.lower() == "t":
            self._toggle_train_mode()
        elif key == 26:  # Ctrl+Z (^Z)
            self.dispatch_action("undo")
        elif key == 25:  # Ctrl+Y (^Y)
            self.dispatch_action("redo")

    # ------------------------------------------------------------------
    def _apply_gesture(self, hands: list[Hand], gesture: GestureEvent, now: float) -> None:
        name = gesture.name
        if name == "none":
            self.last_gesture = "none"
            self.last_confidence = 0.0
            return
        self.last_gesture = name
        self.last_confidence = gesture.confidence
        if self.mode == MODE_TRAIN:
            self._train_tick(hands)
            return

        if name == "pinch":
            self._gesture_pinch(hands, gesture, now)
        elif name == "fist":
            self._gesture_fist(gesture)
        elif name == "open_palm":
            self._gesture_open_palm()
        elif name == "two_hand_pinch":
            self._gesture_two_hand_scale(hands)
        elif name == "twist":
            self._gesture_twist(hands)
        elif name == "point":
            if self.mode == MODE_DRAW:
                self._draw_tick(hands)
        elif name == "swipe_left":
            self.dispatch_action("undo")
            self._log("gesture", "swipe_left")
        elif name == "swipe_right":
            self.dispatch_action("redo")
            self._log("gesture", "swipe_right")
        elif name == "thumbs_up":
            self._gesture_thumbs_up(now)
        elif name == "ok_sign":
            self.dispatch_action("calibrate")
        elif name == "peace":
            self.cycle_primitive()

    # --- specific gesture handlers -------------------------------------
    def _gesture_pinch(self, hands, gesture, now) -> None:
        if len(hands) == 0:
            return
        hand = hands[0]
        idx_tip = hand.landmark(LM.INDEX_TIP)
        if idx_tip is None:
            return
        w, h = self._frame_dims
        px = int(idx_tip.x * w)
        py = int(idx_tip.y * h)
        self._log("gesture", "pinch", 0, x=idx_tip.x, y=idx_tip.y)

        if self._grabbed_id is None:
            # Nothing grabbed -> attempt to grab nearest object by screen point.
            self._try_grab_at(px, py)
            if self._grabbed_id is None:
                # Otherwise spawn a new object at the fingertip.
                self.action_spawn()
        else:
            # Drag the grabbed object following the fingertip (screen-space).
            self._drag_grabbed(idx_tip)

    def _gesture_fist(self, gesture) -> None:
        # A fist locks/grab selection, but pinch is the primary grab. We keep
        # fist as "hold to grab the selected object" for the G fallback parity.
        if not self._grab_held:
            self._grab_selected_or_first()
            self._grab_held = True

    def _gesture_open_palm(self) -> None:
        if self._grabbed_id is not None:
            self._release_grabbed()
        self._grab_held = False

    def _gesture_two_hand_scale(self, hands) -> None:
        if len(hands) < 2:
            return
        g0, g1 = hands[0], hands[1]
        span = self._hand_span_px(g0, g1)
        self._two_hand_span = span
        if self._prev_two_hand_span is not None and self._prev_two_hand_span > 0:
            ratio = span / self._prev_two_hand_span
            if 0.85 < ratio < 1.2:
                self._scale_selected(ratio)
        self._prev_two_hand_span = span

    def _gesture_twist(self, hands) -> None:
        if len(hands) < 2:
            return
        g0, g1 = hands[0], hands[1]
        a0, a1 = self._hand_angle(g0), self._hand_angle(g1)
        angle = a1 - a0
        if self._prev_twist_angle is not None:
            diff = angle - self._prev_twist_angle
            if abs(diff) > 0.02:
                self._rotate_selected(math.degrees(diff) * 30.0)
        self._prev_twist_angle = angle

    def _gesture_thumbs_up(self, now: float) -> None:
        if self._arm_ok_t0 is None:
            self._arm_ok_t0 = now
        elif now - self._arm_ok_t0 >= 1.0:
            self.action_save_report()
            self._arm_ok_t0 = None

    # --- draw mode -----------------------------------------------------
    def _draw_tick(self, hands) -> None:
        if not hands:
            return
        hand = hands[0]
        ip = hand.landmark(LM.INDEX_TIP)
        if ip is None:
            return
        w, h = self._frame_dims
        pt = (int(ip.x * w), int(ip.y * h))
        if self._last_draw_pt:
            self._draw_canvas = self._draw_line(self._draw_canvas, self._last_draw_pt, pt)
        self._last_draw_pt = pt

    def _draw_line(self, canvas, p0, p1):
        import cv2

        color = tuple(self.config.get("air_draw.canvas_color", [80, 200, 255]))
        thickness = self.config.get("air_draw.line_width", 3)
        cv2.line(canvas, p0, p1, color, thickness, lineType=cv2.LINE_AA)
        return canvas

    def _draw_canvas_overlay(self, frame) -> None:
        import cv2

        if self._draw_canvas is not None and self.mode == MODE_DRAW:
            alpha = 0.35
            frame[:] = cv2.addWeighted(
                self._draw_canvas, alpha, frame, 1.0 - alpha
            )

    def _toggle_draw_mode(self) -> None:
        if self.mode == MODE_DRAW:
            self._leave_draw_mode(save=True)
        else:
            self.mode = MODE_DRAW
            w, h = self._frame_dims
            self._draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)
            self._last_draw_pt = None
            log.info("Air-draw mode ON")

    def _leave_draw_mode(self, save: bool) -> None:
        self.mode = MODE_NORMAL
        self._last_draw_pt = None
        if save and self._draw_canvas is not None:
            import cv2

            save_dir = Path(self.config.get("air_draw.save_dir", "output/drawings"))
            save_dir.mkdir(parents=True, exist_ok=True)
            from datetime import datetime

            name = f"drawing_{datetime.now():%Y%m%d_%H%M%S}.png"
            cv2.imwrite(str(save_dir / name), self._draw_canvas)
            self._log("air_draw_saved", name)
            log.info("Saved drawing to %s", save_dir / name)
        self._draw_canvas = None

    # --- grab / drag / release ----------------------------------------
    def _try_grab_at(self, px: int, py: int) -> None:
        best = None
        best_d = float("inf")
        w, h = self._frame_dims
        for obj in self.scene.objects:
            proj = self._world_to_screen(obj.position)
            if proj is None:
                continue
            d = math.hypot(proj[0] - px, proj[1] - py)
            if d < best_d and d < 120:
                best_d = d
                best = obj
        if best is not None:
            self._grabbed_id = best.obj_id
            self.scene.select(best.obj_id)
            best.body.dynamic = False

    def _grab_selected_or_first(self) -> None:
        obj = self.scene.selected() or self.scene.last()
        if obj is not None:
            self._grabbed_id = obj.obj_id
            obj.body.dynamic = False

    def _drag_grabbed(self, idx_tip) -> None:
        obj = self.scene.get(self._grabbed_id)
        if obj is None:
            self._grabbed_id = None
            return
        # Map fingertip to a world-space X/Y that follows the hand, keep a
        # comfortable Z, and estimate a stable Z from the hand span.
        w, h = self._frame_dims
        world_xy = self._screen_delta_to_world(idx_tip.x, idx_tip.y)
        est_z = self._estimate_object_z()
        if est_z is not None:
            obj.position[2] = est_z
        obj.position[0] = world_xy[0]
        obj.position[1] = world_xy[1]
        # Track history for release velocity.
        self.body_cacher.push(obj.obj_id, obj.position.copy())

    def _screen_delta_to_world(self, nx: float, ny: float) -> tuple[float, float]:
        # Convert normalized screen coords to approximate world X/Y within a
        # view box around the origin.
        wx = (nx - 0.5) * 8.0
        wy = (0.5 - ny) * 6.0
        return wx, wy

    def _estimate_object_z(self) -> float | None:
        # Use the most recent hand span to convert to a metric Z.
        if self.depth.is_calibrated and self._palm_spans:
            span_px = self._palm_spans[-1]
            z = self.depth.z_from_span(span_px)
            if z is not None and z > 0:
                return -(z - self.depth.reference_distance_m) * 5.0
        return None

    def _world_to_screen(self, world: np.ndarray):
        if self.renderer is None:
            return None
        try:
            return self.renderer.project(world)
        except Exception:
            return None

    def _release_grabbed(self) -> None:
        obj = self.scene.get(self._grabbed_id)
        if obj is not None:
            vel = self.body_cacher.velocity(obj.obj_id, 1 / 30.0)
            obj.body.dynamic = True
            obj.body.velocity = vel * 0.4
            obj.body.velocity[1] += 1.2  # slight upward release
            self.macro_recorder.record("release", obj_id=obj.obj_id)
            self._log("release", "grabbed", obj.obj_id)
        self._grabbed_id = None
        self._grab_held = False

    def _toggle_grab(self) -> None:
        if self._grabbed_id is None:
            self._grab_selected_or_first()
        else:
            self._release_grabbed()

    # --- scaling / rotation -------------------------------------------
    def _scale_selected(self, factor: float) -> None:
        obj = self.scene.selected() or self.scene.get(self._grabbed_id or -1) or self.scene.last()
        if obj is None:
            return
        self.scene.scale_by(obj.obj_id, factor)
        self.macro_recorder.record("scale", obj_id=obj.obj_id, factor=round(factor, 4))
        self._log("scale", f"f={factor:.2f}", obj.obj_id)

    def _rotate_selected(self, deg: float) -> None:
        obj = self.scene.selected() or self.scene.get(self._grabbed_id or -1) or self.scene.last()
        if obj is None:
            return
        self.scene.rotate(obj.obj_id, "y", deg)
        self.macro_recorder.record("rotate", obj_id=obj.obj_id, deg=round(deg, 3))
        self._log("rotate", f"deg={deg:.1f}", obj.obj_id)

    # --- helpers -------------------------------------------------------
    def _hand_span_px(self, g0: Hand, g1: Hand) -> float:
        if self._frame_dims[0] == 0:
            return 1.0
        w = self._frame_dims[0]
        a0 = g0.landmark(LM.WRIST)
        a1 = g1.landmark(LM.WRIST)
        if a0 is None or a1 is None:
            return self._prev_two_hand_span or 1.0
        return math.hypot((a0.x - a1.x) * w, (a0.y - a1.y) * w)

    def _hand_angle(self, hand: Hand) -> float:
        a = hand.landmark(LM.WRIST)
        b = hand.landmark(LM.MIDDLE_MCP)
        if a is None or b is None:
            return 0.0
        return math.atan2(b.y - a.y, b.x - a.x)

    def _palm_span_px(self, hand: Hand) -> float:
        return self.depth.pan_span_px(hand, self._frame_dims[0] or 1)

    def _start_calibration(self) -> None:
        self._calibrate_until = time.time() + 2.5
        self._palm_spans.clear()
        self._log("calibrate", "started")

    def _calibration_state(self) -> str:
        if time.time() < self._calibrate_until:
            return "collecting…"
        if self.depth.is_calibrated:
            return f"OK (f={self.depth.focal_length_px:.0f}px)"
        return "not calibrated"

    def _scene_info(self) -> str:
        counts = {}
        for o in self.scene.objects:
            counts[o.kind] = counts.get(o.kind, 0) + 1
        parts = [f"{k}:{v}" for k, v in counts.items()]
        return f"objects={len(self.scene.objects)} ({', '.join(parts) or 'none'})"

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------
    def _step_physics(self, dt: float) -> None:
        for obj in self.scene.objects:
            if obj.body.dynamic:
                self.physics.step(obj.body, dt)
                obj.position = obj.body.position.copy()
        if self.config.get("physics.sphere_collision", True):
            self.physics.collide_spheres([o.body for o in self.scene.objects])

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render_scene_frame(self, frame) -> np.ndarray:
        if isinstance(self.renderer, OpenCVRenderer):
            self.renderer.begin_frame(frame)
            for obj in self.scene.objects:
                self.renderer.draw_object(obj)
            base = self.renderer.finish()
            return base
        # GL renderer: draw scene and return a blank placeholder composited
        # with the camera frame for the HUD (GL has its own window).
        if self.renderer is not None:
            self.renderer.begin_frame(None)
            for obj in self.scene.objects:
                self.renderer.draw_object(obj)
            self.renderer.finish()
        return frame

    def _draw_hand_overlay(self, frame, hands) -> None:
        import cv2

        for hand in hands:
            color = hand.color
            pts = [(int(round(hand.landmark(i).x * self._frame_dims[0])),
                    int(round(hand.landmark(i).y * self._frame_dims[1])))
                   for i in range(21) if hand.landmark(i) is not None]
            for p in pts:
                cv2.circle(frame, p, 4, color, -1)
            for a, b in self._hand_connections():
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], color, 1)

    def _hand_connections(self):
        from .vision.hand_tracker import HAND_CONNECTIONS

        return HAND_CONNECTIONS

    def _present(self, frame) -> None:
        import cv2

        if isinstance(self.renderer, OpenCVRenderer):
            cv2.imshow(self.config.get("app.window_title", "GestureForge"), frame)

    # ------------------------------------------------------------------
    # Voice / macro / trainer
    # ------------------------------------------------------------------
    def _drain_voice(self) -> None:
        if self.voice is None:
            return
        for action in self.voice.drain():
            self.dispatch_action(action)
            self._log("voice", action)

    def _toggle_macro_record(self) -> None:
        if self.macro_recorder.recording:
            events = self.macro_recorder.stop()
            save_dir = Path(self.config.get("macros.save_dir", "output/macros"))
            path = self.macro_recorder.save(save_dir / self.config.get("macros.default_name", "macro.json"))
            log.info("Recorded %d macro events -> %s", len(events), path)
            self._log("macro_recorded", str(path))
        else:
            self.macro_recorder.start()
            log.info("Macro recording started")

    def _toggle_macro_replay(self) -> None:
        if self.macro_player.playing:
            self.macro_player.stop()
            return
        path = Path(self.config.get("macros.save_dir", "output/macros")) / self.config.get("macros.default_name", "macro.json")
        if not path.is_file():
            log.warning("No macro found at %s", path)
            return
        if self.macro_player.load(path):
            self.macro_player.start()
            log.info("Replaying macro: %s", path)

    def _toggle_train_mode(self) -> None:
        if self.mode == MODE_TRAIN:
            self.mode = MODE_NORMAL
            self.trainer.stop_recording()
            return
        self.mode = MODE_TRAIN
        if not self.trainer.capture.samples:
            self.trainer.start_recording("custom_gesture_A")
        log.info("Training mode ON — perform gesture to capture samples")

    def _train_tick(self, hands) -> None:
        if not hands:
            return
        self.trainer.capture_frame(hands[0])
        label = self.trainer.current_label or "custom_gesture_A"
        n = self.trainer.count_for(label)
        if n >= self.trainer.samples_per_label:
            err = self.trainer.train()
            self.mode = MODE_NORMAL
            self.trainer.stop_recording()
            log.info("Training complete: %s", err or "ok")
            self._log("train_done", err or "ok")

    # ------------------------------------------------------------------
    # Reports / export
    # ------------------------------------------------------------------
    def action_save_report(self) -> None:
        self._save_scene_and_report()

    def _save_scene_and_report(self) -> None:
        # Scene export (.obj) first.
        try:
            export_dir = Path(self.config.get("export.obj_dir", "output/scenes"))
            export_dir.mkdir(parents=True, exist_ok=True)
            name = self.config.get("export.default_filename", "scene.obj")
            export_scene(self.scene.objects, export_dir / name)
            self._log("scene_export", name)
        except Exception as exc:
            log.warning("Scene export failed: %s", exc)
        # Session analytics report.
        self._generate_report()

    def _generate_report(self) -> None:
        try:
            events = self.session.retrieve_events()
            self.session.save_scene_snapshot(self.scene.objects)
            out = generate_report(
                events,
                self.session.duration(),
                report_dir=self.config.get("analytics.report_dir", "output/reports"),
                heatmap_size=self.config.get("analytics.heatmap_size", 32),
            )
            log.info("Session report written: %s", out)
            self._log("report", str(out.get("pdf") or out.get("png") or out))
        except Exception as exc:
            log.warning("Report generation failed: %s", exc)

    def _session_summary(self) -> None:
        log.info(
            "Session summary: %d objects, %d events, %.1fs",
            len(self.scene.objects),
            self.session._event_count(),
            self.session.duration(),
        )
        if not self.headless_log_only:
            self._generate_report()

    def _log(self, event_type: str, detail: str = "", obj_id=None, x=None, y=None) -> None:
        self.session.log_event(event_type, detail, obj_id or 0, x=x, y=y)

    # ------------------------------------------------------------------
    def _shutdown_cleanup(self) -> None:
        if self.tracker is not None:
            self.tracker.stop()
        if self.voice is not None:
            self.voice.stop()
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
        import cv2

        cv2.destroyAllWindows()
        if hasattr(self, "_cap"):
            try:
                self._cap.release()
            except Exception:
                pass
        self.session.close()
