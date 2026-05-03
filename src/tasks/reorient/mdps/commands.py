from __future__ import annotations

import math as _math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.visualization_markers import VisualizationMarkers
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class InHandReOrientationCommand(CommandTerm):
    """Command term that generates 3D pose commands for in-hand manipulation task.

    This command term generates 3D orientation commands for the object. The orientation commands
    are sampled uniformly from the 3D orientation space. The position commands are the default
    root state of the object.

    The constant position commands is to encourage that the object does not move during the task.
    For instance, the object should not fall off the robot's palm.

    Unlike typical command terms, where the goals are resampled based on time, this command term
    does not resample the goals based on time. Instead, the goals are resampled when the object
    reaches the goal orientation. The goal orientation is considered to be reached when the
    orientation error is below a certain threshold.
    """

    cfg: InHandReOrientationCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: InHandReOrientationCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # object
        self.object: RigidObject = env.scene[cfg.asset_name]

        # create buffers to store the command
        # -- command: (x, y, z)
        self.init_pos_offset = torch.tensor(
            cfg.init_pos_offset, dtype=torch.float, device=self.device
        )
        self.pos_command_w = self.object.data.root_pos_w.clone() + self.init_pos_offset
        self.pos_command_e = self.pos_command_w - self._env.scene.env_origins

        # -- orientation: (w, x, y, z)
        self.quat_command_w = self.object.data.root_quat_w.clone()

        # -- unit vectors
        self._X_UNIT_VEC = torch.tensor([1.0, 0, 0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self._Y_UNIT_VEC = torch.tensor([0, 1.0, 0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self._Z_UNIT_VEC = torch.tensor([0, 0, 1.0], device=self.device).repeat(
            (self.num_envs, 1)
        )

        # -- metrics
        self.metrics["orientation_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["consecutive_success"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self._goal_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._new_success_this_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.random_range = cfg.random_range

    def __str__(self) -> str:
        msg = "InHandManipulationCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired goal pose in the environment frame. Shape is (num_envs, 7)."""
        return torch.cat((self.pos_command_e, self.quat_command_w), dim=-1)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        # -- compute the orientation error
        self.metrics["orientation_error"] = math_utils.quat_error_magnitude(
            self.object.data.root_quat_w, self.quat_command_w
        )
        # -- compute the position error
        self.metrics["position_error"] = torch.norm(
            self.object.data.root_pos_w - self.pos_command_w, dim=1
        )
        successes = self.metrics["orientation_error"] < self.cfg.orientation_success_threshold
        if self.cfg.use_position_success:
            successes = successes & (self.metrics["position_error"] < self.cfg.position_success_threshold)
        # only count success once per goal command
        new_success = successes & ~self._goal_succeeded
        self._new_success_this_step = new_success  # saved before _goal_succeeded update for _update_command
        self.metrics["consecutive_success"] += new_success.float()
        # used for consecutive_success, only record one success per goal command.
        self._goal_succeeded |= successes

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        idx: slice | Sequence[int] = slice(None) if env_ids is None else env_ids
        self.pos_command_w[idx] = self.object.data.root_pos_w[idx] + self.init_pos_offset
        self.pos_command_e[idx] = self.pos_command_w[idx] - self._env.scene.env_origins[idx]
        self.quat_command_w[idx] = self.object.data.root_quat_w[idx]
        self._hold_counter[idx] = 0
        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        self._goal_succeeded[env_ids] = False
        self._hold_counter[env_ids] = 0
        n = len(env_ids)
        min_rad, max_rad = self.random_range
        # approximaly uniformally sample from SO3 around the current command pose
        # Haar-correct: p(θ) ∝ sin²(θ/2); CDF = θ/2 - sin(θ)/2
        lo = 0.5 * (min_rad - _math.sin(min_rad))
        hi = 0.5 * (max_rad - _math.sin(max_rad))
        u = torch.rand((n,), device=self.device) * (hi - lo) + lo
        theta = (12 * u).clamp(min=1e-8).pow(1 / 3)  # cubic-root init
        f = 0.5 * (theta - torch.sin(theta)) - u
        df = 0.5 * (1 - torch.cos(theta))
        angle = (theta - f / df.clamp(min=1e-6)).clamp(min_rad, max_rad)
        axis = torch.randn((n, 3), device=self.device)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        quat_delta = math_utils.quat_from_angle_axis(angle, axis)

        # apply delta to the current object orientation so new goals stay within
        # random_range of the actual object state (not the previous command, which
        # may be unachieved in time-based resampling).
        init_quat = self.object.data.root_quat_w[env_ids]
        quat = math_utils.quat_mul(init_quat, quat_delta)

        self.quat_command_w[env_ids] = (
            math_utils.quat_unique(quat) if self.cfg.make_quat_unique else quat
        )

    def _update_command(self):
        if self.cfg.resample_on == "success":
            successes = self.metrics["orientation_error"] < self.cfg.orientation_success_threshold
            if self.cfg.use_position_success:
                successes = successes & (
                    self.metrics["position_error"] < self.cfg.position_success_threshold
                )

            hold_steps = self.cfg.hold_steps_on_success
            prev_holding = self._hold_counter > 0

            # Decrement hold counters for envs currently in hold phase
            self._hold_counter[prev_holding] -= 1

            # Envs whose hold just expired → resample now
            hold_done = prev_holding & (self._hold_counter == 0)

            if hold_steps > 0:
                # _new_success_this_step was captured in _update_metrics() before _goal_succeeded
                # was updated — this is the only reliable way to detect the first-success edge,
                # since _goal_succeeded is already True by the time _update_command() runs.
                new_first_success = self._new_success_this_step & ~prev_holding
                self._hold_counter[new_first_success] = hold_steps
                # all other successes (post-reset stale case, or goal lost/regained) → immediate resample
                immediate_resample = successes & ~prev_holding & ~new_first_success
            else:
                # hold_steps == 0: original behavior — immediate resample on any success
                immediate_resample = successes & ~prev_holding

            resample_ids = (hold_done | immediate_resample).nonzero(as_tuple=False).squeeze(-1)
            if len(resample_ids) > 0:
                self._resample(resample_ids)
        # "time" mode: base-class timer in compute() calls _resample() automatically

    def _set_debug_vis_impl(self, debug_vis: TYPE_CHECKING):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(
                    self.cfg.goal_pose_visualizer_cfg
                )
            if not hasattr(self, "current_pose_visualizer"):
                self.current_pose_visualizer = VisualizationMarkers(
                    self.cfg.current_pose_visualizer_cfg
                )
            # set visibility
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
            if hasattr(self, "current_pose_visualizer"):
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Goal pose visualization
        # add an offset to the marker position to visualize the goal
        marker_pos = self.pos_command_w + torch.tensor(
            self.cfg.marker_pos_offset, device=self.device
        )
        marker_quat = self.quat_command_w
        # visualize the goal marker
        self.goal_pose_visualizer.visualize(
            translations=marker_pos, orientations=marker_quat
        )

        # Current object pose visualization
        # visualize at actual object position
        current_pos = self.object.data.root_pos_w
        current_quat = self.object.data.root_quat_w
        self.current_pose_visualizer.visualize(
            translations=current_pos, orientations=current_quat
        )


@configclass
class InHandReOrientationCommandCfg(CommandTermCfg):
    """Configuration for the uniform 3D orientation command term.

    Please refer to the :class:`InHandReOrientationCommand` class for more details.
    """

    random_range: tuple[float, float] = (0.0, 1.0)
    """Min/max geodesic angle in radians for goal orientation sampling."""

    class_type: type = InHandReOrientationCommand
    resampling_time_range: tuple[float, float] = (
        1e6,
        1e6,
    )  # no resampling based on time

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    init_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset of the asset from its default position.

    This is used to account for the offset typically present in the object's default position
    so that the object is spawned at a height above the robot's palm. When the position command
    is generated, the object's default position is used as the reference and the offset specified
    is added to it to get the desired position of the object.
    """

    make_quat_unique: bool = MISSING
    """Whether to make the quaternion unique or not.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    orientation_success_threshold: float = MISSING
    """Threshold for the orientation error to consider the goal orientation to be reached."""

    use_position_success: bool = False
    """Whether to also require position error below threshold for success. Defaults to False."""

    position_success_threshold: float = 0.05
    """Threshold for the position error (m) when use_position_success is True."""

    hold_steps_on_success: int = 0
    """Number of timesteps to hold at the goal after first success before resampling.
    0 (default) = immediate resample on success (original behavior)."""

    resample_on: Literal["success", "time"] = "success"
    """When to resample the goal command.

    - ``"success"``: resample when the object reaches the goal (orientation error, and optionally
      position error, fall below their respective thresholds).
    - ``"time"``: resample on a timer; set :attr:`resampling_time_range` to the desired interval.
      The base-class timer handles resampling automatically.
    """

    marker_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset of the marker from the object's desired position.

    This is useful to position the marker at a height above the object's desired position.
    Otherwise, the marker may occlude the object in the visualization.
    """

    # Goal pose visualization - XYZ frame axes (5cm scale)
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_marker",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),  # ~5cm axes
            ),
        },
    )
    """The configuration for the goal pose visualization marker. Defaults to XYZ axes (5cm)."""

    # Current object pose visualization - XYZ frame axes (5cm scale)
    current_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/current_marker",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),  # ~5cm axes
            ),
        },
    )
    """The configuration for the current object pose visualization marker."""
