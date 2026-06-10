from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.envs.mdp.events import (
    randomize_rigid_body_mass,
    randomize_rigid_body_material,
    randomize_rigid_body_scale,
)
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, sample_uniform

from .contacts import contacts


class randomize_rigid_body_material_and_record(randomize_rigid_body_material):
    """Randomize rigid body material and cache applied properties for observations."""

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
        record_key: str | None = None,
    ):
        super().__call__(
            env,
            env_ids,
            static_friction_range,
            dynamic_friction_range,
            restitution_range,
            num_buckets,
            asset_cfg,
            make_consistent,
        )
        key = record_key or f"{asset_cfg.name}_material_properties"
        env.extras[key] = self.asset.root_physx_view.get_material_properties().to(
            device=env.device, dtype=torch.float32
        )


class randomize_rigid_body_mass_and_record(randomize_rigid_body_mass):
    """Randomize rigid body mass and cache applied masses for observations."""

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        mass_distribution_params: tuple[float, float],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
        recompute_inertia: bool = True,
        min_mass: float = 1e-6,
        record_key: str | None = None,
    ):
        super().__call__(
            env,
            env_ids,
            asset_cfg,
            mass_distribution_params,
            operation,
            distribution,
            recompute_inertia,
            min_mass,
        )
        key = record_key or f"{asset_cfg.name}_mass"
        env.extras[key] = self.asset.root_physx_view.get_masses().to(
            device=env.device, dtype=torch.float32
        )


def randomize_rigid_body_scale_and_record(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float] | dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
    relative_child_path: str | None = None,
    key: str = "object_scale",
):
    """Randomize rigid body scale and store applied per-env values for critic observations."""
    randomize_rigid_body_scale(env, env_ids, scale_range, asset_cfg, relative_child_path)

    asset: RigidObject = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids_cpu = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids_cpu = env_ids.cpu()

    if key not in env.extras or env.extras[key].shape != (env.num_envs, 3):
        env.extras[key] = torch.ones((env.num_envs, 3), device=env.device)

    child_path = relative_child_path or ""
    if child_path and not child_path.startswith("/"):
        child_path = "/" + child_path

    stage = get_current_stage()
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)
    for env_id in env_ids_cpu.tolist():
        prim = stage.GetPrimAtPath(prim_paths[env_id] + child_path)
        scale_attr = prim.GetAttribute("xformOp:scale") if prim.IsValid() else None
        scale_value = scale_attr.Get() if scale_attr is not None else None
        if scale_value is None:
            scale_tensor = torch.ones(3, device=env.device)
        else:
            scale_tensor = torch.tensor(scale_value, device=env.device, dtype=torch.float32)
        env.extras[key][env_id] = scale_tensor


class reset_joints_within_limits_range(ManagerTermBase):
    """Reset an articulation's joints to a random position in the given limit ranges.

    This function samples random values for the joint position and velocities from the given limit ranges.
    The values are then set into the physics simulation.

    The parameters to the function are:

    * :attr:`position_range` - a dictionary of position ranges for each joint. The keys of the dictionary are the
      joint names (or regular expressions) of the asset.
    * :attr:`velocity_range` - a dictionary of velocity ranges for each joint. The keys of the dictionary are the
      joint names (or regular expressions) of the asset.
    * :attr:`use_default_offset` - a boolean flag to indicate if the ranges are offset by the default joint state.
      Defaults to False.
    * :attr:`asset_cfg` - the configuration of the asset to reset. Defaults to the entity named "robot" in the scene.
    * :attr:`operation` - whether the ranges are scaled values of the joint limits, or absolute limits.
       Defaults to "abs".

    The dictionary values are a tuple of the form ``(a, b)``. Based on the operation, these values are
    interpreted differently:

    * If the operation is "abs", the values are the absolute minimum and maximum values for the joint, i.e.
      the joint range becomes ``[a, b]``.
    * If the operation is "scale", the values are the scaling factors for the joint limits, i.e. the joint range
      becomes ``[a * min_joint_limit, b * max_joint_limit]``.

    If the ``a`` or the ``b`` value is ``None``, the joint limits are used instead.

    Note:
        If the dictionary does not contain a key, the joint position or joint velocity is set to the default value for
        that joint.

    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # check if the cfg has the required parameters
        if "position_range" not in cfg.params or "velocity_range" not in cfg.params:
            raise ValueError(
                "The term 'reset_joints_within_range' requires parameters: 'position_range' and 'velocity_range'."
                f" Received: {list(cfg.params.keys())}."
            )

        # parse the parameters
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        use_default_offset = cfg.params.get("use_default_offset", False)
        operation = cfg.params.get("operation", "abs")
        # check if the operation is valid
        if operation not in ["abs", "scale"]:
            raise ValueError(
                f"For event 'reset_joints_within_limits_range', unknown operation: '{operation}'."
                " Please use 'abs' or 'scale'."
            )

        # extract the used quantities (to enable type-hinting)
        self._asset: Articulation = env.scene[asset_cfg.name]
        default_joint_pos = self._asset.data.default_joint_pos[0]
        default_joint_vel = self._asset.data.default_joint_vel[0]

        # create buffers to store the joint position range
        self._pos_ranges = self._asset.data.soft_joint_pos_limits[0].clone()
        # parse joint position ranges
        pos_joint_ids = []
        for joint_name, joint_range in cfg.params["position_range"].items():
            # find the joint ids
            joint_ids = self._asset.find_joints(joint_name)[0]
            pos_joint_ids.extend(joint_ids)

            # set the joint position ranges based on the given values
            if operation == "abs":
                if joint_range[0] is not None:
                    self._pos_ranges[joint_ids, 0] = joint_range[0]
                if joint_range[1] is not None:
                    self._pos_ranges[joint_ids, 1] = joint_range[1]
            elif operation == "scale":
                if joint_range[0] is not None:
                    self._pos_ranges[joint_ids, 0] *= joint_range[0]
                if joint_range[1] is not None:
                    self._pos_ranges[joint_ids, 1] *= joint_range[1]
            else:
                raise ValueError(
                    f"Unknown operation: '{operation}' for joint position ranges. Please use 'abs' or 'scale'."
                )
            # add the default offset
            if use_default_offset:
                self._pos_ranges[joint_ids] += default_joint_pos[joint_ids].unsqueeze(1)

        # store the joint pos ids (used later to sample the joint positions)
        self._pos_joint_ids = torch.tensor(
            pos_joint_ids, device=self._pos_ranges.device
        )
        self._pos_ranges = self._pos_ranges[self._pos_joint_ids]

        # create buffers to store the joint velocity range
        self._vel_ranges = torch.stack(
            [
                -self._asset.data.soft_joint_vel_limits[0],
                self._asset.data.soft_joint_vel_limits[0],
            ],
            dim=1,
        )
        # parse joint velocity ranges
        vel_joint_ids = []
        for joint_name, joint_range in cfg.params["velocity_range"].items():
            # find the joint ids
            joint_ids = self._asset.find_joints(joint_name)[0]
            vel_joint_ids.extend(joint_ids)

            # set the joint position ranges based on the given values
            if operation == "abs":
                if joint_range[0] is not None:
                    self._vel_ranges[joint_ids, 0] = joint_range[0]
                if joint_range[1] is not None:
                    self._vel_ranges[joint_ids, 1] = joint_range[1]
            elif operation == "scale":
                if joint_range[0] is not None:
                    self._vel_ranges[joint_ids, 0] = (
                        joint_range[0] * self._vel_ranges[joint_ids, 0]
                    )
                if joint_range[1] is not None:
                    self._vel_ranges[joint_ids, 1] = (
                        joint_range[1] * self._vel_ranges[joint_ids, 1]
                    )
            else:
                raise ValueError(
                    f"Unknown operation: '{operation}' for joint velocity ranges. Please use 'abs' or 'scale'."
                )
            # add the default offset
            if use_default_offset:
                self._vel_ranges[joint_ids] += default_joint_vel[joint_ids].unsqueeze(1)

        # store the joint vel ids (used later to sample the joint positions)
        self._vel_joint_ids = torch.tensor(
            vel_joint_ids, device=self._vel_ranges.device
        )
        self._vel_ranges = self._vel_ranges[self._vel_joint_ids]

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        position_range: dict[str, tuple[float | None, float | None]],
        velocity_range: dict[str, tuple[float | None, float | None]],
        use_default_offset: bool = False,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        operation: Literal["abs", "scale"] = "abs",
    ):
        # get default joint state
        joint_pos = self._asset.data.default_joint_pos[env_ids].clone()
        joint_vel = self._asset.data.default_joint_vel[env_ids].clone()

        # sample random joint positions for each joint
        if len(self._pos_joint_ids) > 0:
            joint_pos_shape = (len(env_ids), len(self._pos_joint_ids))
            joint_pos[:, self._pos_joint_ids] = sample_uniform(
                self._pos_ranges[:, 0],
                self._pos_ranges[:, 1],
                joint_pos_shape,
                device=joint_pos.device,
            )
            # clip the joint positions to the joint limits
            joint_pos_limits = self._asset.data.soft_joint_pos_limits[
                0, self._pos_joint_ids
            ]
            joint_pos = joint_pos.clamp(joint_pos_limits[:, 0], joint_pos_limits[:, 1])

        # sample random joint velocities for each joint
        if len(self._vel_joint_ids) > 0:
            joint_vel_shape = (len(env_ids), len(self._vel_joint_ids))
            joint_vel[:, self._vel_joint_ids] = sample_uniform(
                self._vel_ranges[:, 0],
                self._vel_ranges[:, 1],
                joint_vel_shape,
                device=joint_vel.device,
            )
            # clip the joint velocities to the joint limits
            joint_vel_limits = self._asset.data.soft_joint_vel_limits[
                0, self._vel_joint_ids
            ]
            joint_vel = joint_vel.clamp(-joint_vel_limits, joint_vel_limits)

        # set into the physics simulation
        self._asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


class randomize_hand_object_default_pose(ManagerTermBase):
    """Randomize the robot root orientation and rotate the object's default pose with it."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        base_asset_cfg: SceneEntityCfg = cfg.params.get(
            "base_asset_cfg", SceneEntityCfg("robot")
        )
        object_asset_cfg: SceneEntityCfg = cfg.params.get(
            "object_asset_cfg", SceneEntityCfg("object")
        )
        self._robot: Articulation = env.scene[base_asset_cfg.name]
        self._object: RigidObject = env.scene[object_asset_cfg.name]

        if base_asset_cfg.body_names is not None:
            body_ids, _ = self._robot.find_bodies(base_asset_cfg.body_names)
            self._base_body_id: int = body_ids[0]
        else:
            self._base_body_id = 0 # default to the root body

        self.roll_range = cfg.params.get("roll_range", (-torch.pi, torch.pi))
        self.pitch_range = cfg.params.get("pitch_range", (-torch.pi, torch.pi))
        self.yaw_range = cfg.params.get("yaw_range", (-torch.pi, torch.pi))

        # Store original unrotated default states so resets don't accumulate rotations.
        self._orig_robot_default = self._robot.data.default_root_state.clone()
        self._orig_object_default = self._object.data.default_root_state.clone()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        base_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base*"),
        object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        roll_range: tuple[float, float] | None = None,
        pitch_range: tuple[float, float] | None = None,
        yaw_range: tuple[float, float] | None = None,
    ):
        if env_ids is None or env_ids == slice(None):
            env_ids = torch.arange(env.num_envs, device=env.device)

        roll_range = self.roll_range if roll_range is None else roll_range
        pitch_range = self.pitch_range if pitch_range is None else pitch_range
        yaw_range = self.yaw_range if yaw_range is None else yaw_range

        roll = torch.empty(len(env_ids), device=env.device).uniform_(*roll_range)
        pitch = torch.empty(len(env_ids), device=env.device).uniform_(*pitch_range)
        yaw = torch.empty(len(env_ids), device=env.device).uniform_(*yaw_range)
        quat_delta = quat_from_euler_xyz(roll, pitch, yaw)

        env_origins = env.scene.env_origins[env_ids]

        # Always rotate relative to the original unrotated defaults to avoid accumulation.
        robot_states = self._orig_robot_default[env_ids].clone()
        object_states = self._orig_object_default[env_ids].clone()

        # Rigid rotation around the robot root (env frame).
        # body_pos_w is unreliable at startup (not yet initialized), so we use
        # default_root_state which is always valid from config.
        pivot = robot_states[:, :3].clone()
        robot_states[:, 3:7] = math_utils.quat_mul(quat_delta, robot_states[:, 3:7])
        object_states[:, :3] = pivot + quat_apply(quat_delta, object_states[:, :3] - pivot)
        object_states[:, 3:7] = math_utils.quat_mul(quat_delta, object_states[:, 3:7])

        self._robot.data.default_root_state[env_ids] = robot_states
        self._object.data.default_root_state[env_ids] = object_states

        robot_root_states_w = robot_states.clone()
        object_root_states_w = object_states.clone()
        robot_root_states_w[:, :3] += env_origins
        object_root_states_w[:, :3] += env_origins

        self._robot.write_root_state_to_sim(robot_root_states_w, env_ids=env_ids)
        self._object.write_root_state_to_sim(object_root_states_w, env_ids=env_ids)


def _normalize_env_ids(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor | Sequence[int] | slice | None
) -> torch.Tensor:
    """Normalize environment ids to a tensor on the env device."""
    if env_ids is None or env_ids == slice(None):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        start = 0 if env_ids.start is None else env_ids.start
        stop = env.num_envs if env_ids.stop is None else env_ids.stop
        step = 1 if env_ids.step is None else env_ids.step
        return torch.arange(start, stop, step, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


class apply_gravity_compensation_assist(ManagerTermBase):
    """Gravity-compensation assist that decays per step when good contact is detected.

    _assist_scale starts at 1.0 each episode and is multiplied by decay_ratio for each env
    where good_contact_count >= min_good_contacts. It is decay-only (never increases mid-episode).

    Good contact requires both force magnitude > contact_threshold AND force direction within
    contact_pose_range_deg of the fingertip normal (force-direction quality check).
    """

    def __init__(self, cfg: EventTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._assist_scale = torch.ones(env.num_envs, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            self._assist_scale[:] = 1.0
        else:
            self._assist_scale[env_ids] = 1.0

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        contact_threshold: float = 1.0,
        contact_sensor_names: Sequence[str] | None = None,
        contact_pose_range_deg: float = 50.0,
        min_good_contacts: int = 2,
        decay_ratio: float = 0.95,  # 0.95^120 (1s) ~ 0, so decays to 0 in ~1s
    ) -> None:
        env_ids_t = _normalize_env_ids(env, env_ids)
        obj: RigidObject = env.scene[asset_cfg.name]

        # Current physics gravity — live from PhysX after variable_gravity applied.
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        grav_raw = physics_sim_view.get_gravity()

        gravity_vec = torch.tensor(
            [grav_raw[0], grav_raw[1], grav_raw[2]], device=env.device, dtype=torch.float32
        )
        # Actual per-env mass; get_masses() returns CPU tensor.
        masses = obj.root_physx_view.get_masses().to(env.device)  # (num_envs, 1)
        mass = masses[env_ids_t, 0]                               # (n,)

        # Decay wherever force-direction-quality good contact count >= min_good_contacts.
        # contact = (
        #     good_contact_count(env, _FINGERTIP_SENSOR_NAMES,
        #                        force_threshold=contact_threshold,
        #                        contact_pose_range_deg=contact_pose_range_deg)
        #     >= min_good_contacts
        # )                                                          # (num_envs,) bool
        if contact_sensor_names is None:
            contact = contacts(env, threshold=contact_threshold, mode='any')
        else:
            contact_mags = []
            for sensor_name in contact_sensor_names:
                contact_sensor = env.scene.sensors[sensor_name]
                contact_force = contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
                contact_mags.append(torch.norm(contact_force, dim=-1) > contact_threshold)
            contact = torch.stack(contact_mags, dim=-1).sum(dim=-1) >= min_good_contacts
        self._assist_scale[env_ids_t] = torch.where(
            contact[env_ids_t],
            (self._assist_scale[env_ids_t] * decay_ratio).round(decimals=4),
            self._assist_scale[env_ids_t],
        )

        # Apply force; shape (n, 1, 3) required by set_external_force_and_torque.
        scale = self._assist_scale[env_ids_t]
        force_vec = -gravity_vec.unsqueeze(0) * (mass * scale).unsqueeze(-1)
        assist_force = force_vec.unsqueeze(1)
        torques = torch.zeros_like(assist_force)

        obj.set_external_force_and_torque(
            assist_force, torques, env_ids=env_ids_t, is_global=True
        )


class reset_root_state_from_pose(ManagerTermBase):
    """Reset an asset root state to a fixed pose and velocity."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
        self._asset = env.scene[asset_cfg.name]

        pose = cfg.params.get("pose")
        if pose is None or len(pose) != 7:
            raise ValueError(
                "reset_root_state_from_pose requires 'pose' as (x, y, z, w, x, y, z)."
            )

        velocity = cfg.params.get("velocity", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        if len(velocity) != 6:
            raise ValueError(
                "reset_root_state_from_pose requires 'velocity' as (vx, vy, vz, wx, wy, wz)."
            )

        self._pose = torch.tensor(pose, dtype=torch.float32, device=env.device)
        self._velocity = torch.tensor(velocity, dtype=torch.float32, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | Sequence[int] | slice | None,
        pose: tuple[float, float, float, float, float, float, float] | None = None,
        velocity: tuple[float, float, float, float, float, float] | None = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ):
        del asset_cfg
        env_ids_t = _normalize_env_ids(env, env_ids)
        pose_tensor = self._pose if pose is None else torch.tensor(pose, dtype=torch.float32, device=env.device)
        vel_tensor = (
            self._velocity if velocity is None else torch.tensor(velocity, dtype=torch.float32, device=env.device)
        )

        root_states = self._asset.data.default_root_state[env_ids_t].clone()
        root_states[:, 0:3] = pose_tensor[0:3] + env.scene.env_origins[env_ids_t]
        root_states[:, 3:7] = pose_tensor[3:7]
        root_states[:, 7:10] = vel_tensor[0:3]
        root_states[:, 10:13] = vel_tensor[3:6]
        self._asset.write_root_state_to_sim(root_states, env_ids=env_ids_t)


class reset_joints_around_default(ManagerTermBase):
    """Sample candidate joint positions as default plus uniform per-joint deltas."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._joint_reset_delta = float(cfg.params.get("joint_reset_delta", 0.25))

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | Sequence[int] | slice | None,
        joint_reset_delta: float | None = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        del asset_cfg
        env_ids_t = _normalize_env_ids(env, env_ids)
        delta = self._joint_reset_delta if joint_reset_delta is None else float(joint_reset_delta)

        joint_pos = self._asset.data.default_joint_pos[env_ids_t].clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos += delta * (2.0 * torch.rand_like(joint_pos) - 1.0)

        joint_limits = self._asset.data.soft_joint_pos_limits[env_ids_t]
        joint_pos = torch.clamp(joint_pos, joint_limits[..., 0], joint_limits[..., 1])
        self._asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)
