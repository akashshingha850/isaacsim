#!/usr/bin/env python
"""
| File: launch.py
| Description: Config-driven Pegasus standalone launcher. Reads a YAML file
|   (default: config.yaml next to this script) describing the world, vehicle,
|   control backend, and optional uXRCE-DDS agent / RViz processes, runs the sim.
|
| Usage:
|   isaac_run PegasusSimulator/examples/launch.py
|   isaac_run PegasusSimulator/examples/launch.py --config /path/to/other.yaml
"""

import os
import sys
import argparse
import subprocess

import yaml

# -----------------------------------------------------------------------------
# 1. Read the config FIRST (plain YAML, safe before Isaac starts) so we know
#    whether to launch headless. SimulationApp must be created before any other
#    Isaac/omni import, otherwise the simulator crashes.
# -----------------------------------------------------------------------------
def _parse_config():
    parser = argparse.ArgumentParser(description="Config-driven Pegasus launcher")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
        help="Path to the YAML config file.",
    )
    # Ignore any extra args the kit wrapper appends.
    args, _ = parser.parse_known_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}
    print(f"[launch] Loaded config: {args.config}")
    return cfg


CONFIG = _parse_config()

import carb
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": bool(CONFIG.get("app", {}).get("headless", False))})

# -----------------------------------------------------------------------------
# 2. Everything below runs after the simulator is up.
# -----------------------------------------------------------------------------
import numpy as np
import omni.timeline
import omni.usd
from omni.isaac.core.world import World
from scipy.spatial.transform import Rotation
from pxr import UsdGeom, UsdLux, Gf, UsdPhysics

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend,
    PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface


def _resolve_world(world_cfg):
    """Return the USD path of the world to load (custom path wins over preset)."""
    usd_path = world_cfg.get("usd_path")
    if usd_path:
        return usd_path
    preset = world_cfg.get("preset", "Curved Gridroom")
    if preset not in SIMULATION_ENVIRONMENTS:
        carb.log_warn(
            f"[launch] World preset '{preset}' not found; falling back to 'Curved Gridroom'. "
            f"Available: {list(SIMULATION_ENVIRONMENTS)}"
        )
        preset = "Curved Gridroom"
    return SIMULATION_ENVIRONMENTS[preset]


def _resolve_vehicle_usd(vehicle_cfg):
    """Return the USD path of the vehicle model (custom path wins over preset)."""
    usd_path = vehicle_cfg.get("usd_path")
    if usd_path:
        return usd_path
    preset = vehicle_cfg.get("preset", "Iris")
    if preset not in ROBOTS:
        carb.log_warn(
            f"[launch] Vehicle preset '{preset}' not found; falling back to 'Iris'. "
            f"Available: {list(ROBOTS)}"
        )
        preset = "Iris"
    return ROBOTS[preset]


class Launcher:
    """Builds and runs a Pegasus simulation entirely from the YAML config."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._procs = []  # child processes (uXRCE-DDS agent / RViz) to clean up on exit

        # Bring up the Micro XRCE-DDS Agent BEFORE PX4 SITL autolaunches (which
        # happens on timeline.play()), so PX4's uxrce_dds_client has an agent to
        # connect to. The client retries, so exact ordering is not critical.
        self._start_uxrce_agent(cfg.get("uxrce_dds", {}))

        self.timeline = omni.timeline.get_timeline_interface()

        # Start Pegasus + the World singleton.
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # --- Load the world -------------------------------------------------
        world_cfg = cfg.get("world", {})
        world_usd = _resolve_world(world_cfg)
        print(f"[launch] Loading world: {world_usd}")
        self.pg.load_environment(world_usd)

        # load_environment() is async: pump updates until the layout is composed.
        for _ in range(2000):
            simulation_app.update()
            if self.world.stage.GetPrimAtPath("/World/layout").IsValid():
                break

        if world_cfg.get("add_ground_plane", False):
            self._add_ground_plane(float(world_cfg.get("ground_z", 0.0)))

        # --- Build the vehicle + backend ------------------------------------
        vehicle_cfg = cfg.get("vehicle", {})
        config_multirotor = MultirotorConfig()
        config_multirotor.backends = self._build_backends(cfg.get("backend", {}), vehicle_cfg)

        vehicle_usd = _resolve_vehicle_usd(vehicle_cfg)
        pos = vehicle_cfg.get("position", [0.0, 0.0, 0.15])
        euler = vehicle_cfg.get("orientation_euler_deg", [0.0, 0.0, 0.0])
        print(f"[launch] Spawning vehicle: {vehicle_usd} at {pos}")
        self.vehicle = Multirotor(
            vehicle_cfg.get("stage_prefix", "/World/quadrotor"),
            vehicle_usd,
            int(vehicle_cfg.get("vehicle_id", 0)),
            pos,
            Rotation.from_euler("XYZ", euler, degrees=True).as_quat(),
            config=config_multirotor,
        )

        # --- Cameras ---------------------------------------------------------
        self._setup_chase_camera(cfg.get("chase_camera", {}))
        self._setup_static_cameras(cfg.get("static_cameras", []))

        # --- Optional external ROS 2 processes ------------------------------
        self._start_rviz(cfg.get("rviz", {}))

        # Initialize articulations.
        print("[launch] Resetting world (initializing physics/articulations)...", flush=True)
        self.world.reset()
        print("[launch] World reset complete — entering run loop.", flush=True)
        self.stop_sim = False

    # ------------------------------------------------------------------ world
    def _add_ground_plane(self, ground_z):
        """A flat, invisible ground collider + dome light for render-only worlds."""
        stage = self.world.stage
        ground = UsdGeom.Cube.Define(stage, "/World/launch_ground")
        ground.GetSizeAttr().Set(2.0)
        half_thickness = 0.5
        half_extent = 200.0
        gx = UsdGeom.Xformable(ground.GetPrim())
        gx.ClearXformOpOrder()
        gx.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, ground_z - half_thickness))
        gx.AddScaleOp().Set(Gf.Vec3f(half_extent, half_extent, half_thickness))
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        UsdGeom.Imageable(ground.GetPrim()).MakeInvisible()
        UsdLux.DomeLight.Define(stage, "/World/launch_dome_light").CreateIntensityAttr(1000.0)
        print(f"[launch] Added flat ground collider at z={ground_z}")

    # ----------------------------------------------------------- chase camera
    def _setup_chase_camera(self, chase_cfg):
        """Third-person camera that tracks the vehicle. Isaac's PhysX follow
        cameras don't update in a standalone SimulationApp (they're driven by
        the GUI editor loop), so we aim a plain camera prim ourselves each frame."""
        self.chase_camera_path = None
        if not chase_cfg.get("enabled", False):
            return
        self.chase_camera_path = chase_cfg.get("stage_path", "/World/chase_camera")
        self.chase_camera_offset = np.array(chase_cfg.get("offset", [-4.0, 0.0, 1.5]), dtype=float)
        self.chase_camera_yaw_follow = bool(chase_cfg.get("yaw_follow", True))
        self.chase_camera_follow_selection = bool(chase_cfg.get("follow_selection", True))
        self.chase_camera_smoothing = min(max(float(chase_cfg.get("smoothing", 0.15)), 0.0), 0.95)
        self.chase_camera_z_min = chase_cfg.get("z_min")
        self.chase_camera_z_max = chase_cfg.get("z_max")
        self._chase_eye = None
        self._chase_target = None
        self._chase_viewport = None
        self._chase_last_target = None
        cam = UsdGeom.Camera.Define(self.world.stage, self.chase_camera_path)
        if chase_cfg.get("focal_length"):
            cam.CreateFocalLengthAttr(float(chase_cfg["focal_length"]))
        print(f"[launch] Chase camera at {self.chase_camera_path} "
              f"(offset={self.chase_camera_offset.tolist()}, yaw_follow={self.chase_camera_yaw_follow}, "
              f"follow_selection={self.chase_camera_follow_selection})")

        if not bool(self.cfg.get("app", {}).get("headless", False)):
            from omni.kit.viewport.utility import get_active_viewport
            self._chase_viewport = get_active_viewport()
            if chase_cfg.get("set_active", True):
                self._chase_viewport.camera_path = self.chase_camera_path
                print("[launch] Viewport switched to the chase camera "
                      "(pick 'Perspective' in the viewport camera dropdown for free mouse control)")

    def _aim_camera(self, cam_path, eye, target):
        """Place a camera prim at `eye` looking at `target` (world frame, +Z up)."""
        eye = Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2]))
        target = Gf.Vec3d(float(target[0]), float(target[1]), float(target[2]))
        f = (target - eye).GetNormalized()              # forward (camera looks down -Z)
        up_ref = Gf.Vec3d(0, 0, 1)
        if abs(Gf.Dot(f, up_ref)) > 0.999:              # straight up/down view: keep +Y ("north") up
            up_ref = Gf.Vec3d(0, 1, 0)
        r = Gf.Cross(f, up_ref).GetNormalized()         # right
        u = Gf.Cross(r, f)                              # up
        # Row-vector transform: rows are the world directions of camera local X, Y, Z.
        m = Gf.Matrix4d(
            r[0],  r[1],  r[2],  0.0,
            u[0],  u[1],  u[2],  0.0,
            -f[0], -f[1], -f[2], 0.0,
            eye[0], eye[1], eye[2], 1.0,
        )
        cam = UsdGeom.Xformable(self.world.stage.GetPrimAtPath(cam_path))
        cam.ClearXformOpOrder()
        cam.AddTransformOp().Set(m)

    def _setup_static_cameras(self, cams_cfg):
        """Fixed viewpoint cameras, selectable from the viewport camera dropdown."""
        for c in cams_cfg or []:
            path = "/World/" + c.get("name", "static_camera")
            cam = UsdGeom.Camera.Define(self.world.stage, path)
            if c.get("focal_length"):
                cam.CreateFocalLengthAttr(float(c["focal_length"]))
            eye = c.get("position", [0.0, 0.0, 2.0])
            target = c.get("look_at", [0.0, 0.0, 0.0])
            self._aim_camera(path, eye, target)
            print(f"[launch] Static camera {path}: {eye} -> {target}")
            # Optionally park the editor Perspective camera at the same pose, so
            # switching the viewport to Perspective starts from a sensible view.
            if c.get("set_perspective", False) and not bool(self.cfg.get("app", {}).get("headless", False)):
                self.pg.set_viewport_camera(eye, target)

    def _selected_prim_pose(self):
        """World position + yaw of the first prim selected in the GUI, or (None, None)
        when nothing usable is selected."""
        for path in omni.usd.get_context().get_selection().get_selected_prim_paths():
            if path == self.chase_camera_path or path.startswith(self.chase_camera_path + "/"):
                continue  # following itself would degenerate the look-at
            prim = self.world.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            m = omni.usd.get_world_transform_matrix(prim)
            pos = np.array(m.ExtractTranslation(), dtype=float)
            x_axis = m.GetRow3(0)  # world direction of the prim's local X (forward)
            yaw = float(np.arctan2(x_axis[1], x_axis[0]))
            return pos, yaw, path
        return None, None, "vehicle"

    def _update_chase_camera(self):
        """Keep the chase camera behind its target. The target is the prim selected
        in the GUI (Gazebo-style object follow) while this camera drives the
        viewport, otherwise the drone. Rotating the body-frame offset by the
        target's yaw keeps the camera behind it; yaw-only keeps the horizon level
        (camera doesn't flip when the drone rolls)."""
        target_pos = target_yaw = None
        target_label = "vehicle"
        if (self.chase_camera_follow_selection and self._chase_viewport is not None
                and str(self._chase_viewport.camera_path) == self.chase_camera_path):
            target_pos, target_yaw, target_label = self._selected_prim_pose()
        if target_pos is None:
            target_pos = np.array(self.vehicle.state.position, dtype=float)  # ENU
            target_yaw = Rotation.from_quat(self.vehicle.state.attitude).as_euler("xyz")[2]
        if target_label != self._chase_last_target:
            self._chase_last_target = target_label
            print(f"[launch] Chase camera following: {target_label}", flush=True)

        if self.chase_camera_yaw_follow:
            world_offset = Rotation.from_euler("z", target_yaw).apply(self.chase_camera_offset)
        else:
            world_offset = self.chase_camera_offset
        eye = target_pos + world_offset
        # Keep the camera inside the building: clamp height (e.g. under the ceiling)
        # even when the target flies high.
        if self.chase_camera_z_max is not None:
            eye[2] = min(eye[2], float(self.chase_camera_z_max))
        if self.chase_camera_z_min is not None:
            eye[2] = max(eye[2], float(self.chase_camera_z_min))

        if self._chase_eye is None or self.chase_camera_smoothing <= 0.0:
            self._chase_eye, self._chase_target = eye, target_pos
        else:
            k = 1.0 - self.chase_camera_smoothing
            self._chase_eye = self._chase_eye + k * (eye - self._chase_eye)
            self._chase_target = self._chase_target + k * (target_pos - self._chase_target)

        if np.linalg.norm(self._chase_target - self._chase_eye) < 1e-6:
            return  # degenerate look-at (eye on top of target)
        self._aim_camera(self.chase_camera_path, self._chase_eye, self._chase_target)

    # ---------------------------------------------------------------- backend
    def _build_backends(self, backend_cfg, vehicle_cfg):
        btype = str(backend_cfg.get("type", "px4")).lower()
        vehicle_id = int(vehicle_cfg.get("vehicle_id", 0))

        if btype == "none":
            print("[launch] Backend: none (no autopilot / ROS control)")
            return []

        if btype == "ros2":
            from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
            r = backend_cfg.get("ros2", {})
            print("[launch] Backend: ROS2Backend")
            return [ROS2Backend(vehicle_id=vehicle_id, config={
                "namespace": r.get("namespace", "drone"),
                "pub_sensors": r.get("pub_sensors", True),
                "pub_graphical_sensors": r.get("pub_graphical_sensors", True),
                "pub_state": r.get("pub_state", True),
                "pub_tf": r.get("pub_tf", False),
                "sub_control": r.get("sub_control", False),
            })]

        # default: PX4 MAVLink
        p = backend_cfg.get("px4", {})
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": vehicle_id,
            "px4_autolaunch": bool(p.get("autolaunch", True)),
            "px4_dir": p.get("px4_dir") or self.pg.px4_path,
            "px4_vehicle_model": p.get("airframe") or self.pg.px4_default_airframe,
            "enable_lockstep": bool(p.get("enable_lockstep", True)),
        })
        print("[launch] Backend: PX4MavlinkBackend")
        return [PX4MavlinkBackend(mavlink_config)]

    # ------------------------------------------------------- external processes
    def _spawn(self, argv, name, env=None):
        try:
            proc = subprocess.Popen(argv, env=env)
            self._procs.append(proc)
            print(f"[launch] Started {name} (pid {proc.pid}): {' '.join(argv)}")
        except FileNotFoundError:
            carb.log_warn(
                f"[launch] Could not start {name}: '{argv[0]}' not found. "
                f"Is it installed and on PATH (or given as an absolute path)?"
            )

    def _start_uxrce_agent(self, uxrce_cfg):
        """Launch the Micro XRCE-DDS Agent that turns PX4's uORB stream into
        /fmu/... ROS 2 topics. UDP transport for PX4 SITL (sim); serial for a
        real FC over the Jetson UART (hardware)."""
        if not uxrce_cfg.get("enabled", False) or not uxrce_cfg.get("autostart_agent", True):
            return
        cmd = uxrce_cfg.get("command")
        if cmd:
            argv = cmd if isinstance(cmd, list) else cmd.split()
        else:
            binary = uxrce_cfg.get("agent_binary", "MicroXRCEDDSAgent")
            transport = str(uxrce_cfg.get("transport", "udp")).lower()
            if transport == "serial":
                argv = [binary, "serial", "--dev",
                        str(uxrce_cfg.get("serial_dev", "/dev/ttyTHS1")),
                        "-b", str(uxrce_cfg.get("serial_baud", 921600))]
            else:  # udp (PX4 SITL default)
                argv = [binary, "udp4", "-p", str(uxrce_cfg.get("udp_port", 8888))]

        # The agent is spawned from INSIDE Isaac's process, which pollutes
        # LD_LIBRARY_PATH with Isaac's own libs (a mismatched spdlog/fmt that
        # breaks the agent with an "undefined symbol" error). Give the agent a
        # SANITIZED env: LD_LIBRARY_PATH = only its own lib/ dir, so its shared
        # objects come from there and system spdlog/fmt resolve via ldconfig.
        env = dict(os.environ)
        binary_path = argv[0]
        if os.path.isabs(binary_path):
            lib_dir = os.path.join(os.path.dirname(os.path.dirname(binary_path)), "lib")
            env["LD_LIBRARY_PATH"] = lib_dir if os.path.isdir(lib_dir) else ""
        else:
            # binary resolved from PATH: still strip Isaac's libs from the agent env
            env["LD_LIBRARY_PATH"] = ""
        self._spawn(argv, "Micro XRCE-DDS Agent", env=env)

    def _start_rviz(self, rviz_cfg):
        if not rviz_cfg.get("enabled", False):
            return
        argv = ["rviz2"]
        if rviz_cfg.get("config"):
            argv += ["-d", rviz_cfg["config"]]
        self._spawn(argv, "RViz")

    # -------------------------------------------------------------------- loop
    def run(self):
        self.timeline.play()
        hb = int(self.cfg.get("app", {}).get("heartbeat_steps", 2000))
        if hb > 0:
            print(f"[launch] timeline.play() called — stepping. Heartbeat every {hb} steps.", flush=True)
        else:
            print("[launch] timeline.play() called — stepping. Heartbeat disabled.", flush=True)
        step = 0
        while simulation_app.is_running() and not self.stop_sim:
            if self.chase_camera_path:
                self._update_chase_camera()
            self.world.step(render=True)
            step += 1
            if hb > 0 and step % hb == 0:
                # sim_time advancing => physics stepping; frozen => render-bound / stalled
                t = self.world.current_time
                pos = getattr(self.vehicle.state, "position", None)
                print(f"[launch] heartbeat step={step} sim_time={t:.2f}s vehicle_pos={pos}", flush=True)
        self._shutdown()

    def _shutdown(self):
        carb.log_warn("[launch] Simulation App is closing.")
        for proc in self._procs:
            proc.terminate()
        self.timeline.stop()
        simulation_app.close()


def main():
    Launcher(CONFIG).run()


if __name__ == "__main__":
    main()
