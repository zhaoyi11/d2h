```python
    track_position = RewTerm(
        func=task_mdp.track_position,
        weight=2.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "object_pose",
            "pos_scale": 2.0,
            "pos_temp": 0.5,
        },
    )

    track_orientation = RewTerm(
        func=task_mdp.track_orientation,
        weight=5.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "object_pose",
            "rot_scale": 5.0,
            "rot_temp": 1.0,
        },
    )
```

```python
def track_orientation(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    rot_scale: float = 5.0,
    rot_temp: float = 1.0,
) -> torch.Tensor:
    """Reward for tracking the object orientation with an exponential (Gaussian) kernel.

    reward = exp(-||active_quat_vec||_2^2 * rot_scale / rot_temp)

    where active_quat = goal_quat * conj(object_quat) is the relative rotation and
    active_quat_vec is its vector (imaginary) part. For a rotation of angle theta,
    ||active_quat_vec||_2 = |sin(theta/2)|, so the reward is 1 at alignment and decays
    smoothly with orientation error.

    The reward is from https://arxiv.org/abs/2509.07445 Gemini-1.5-Flash.
    """
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)

    goal_quat_w = command_term.command[:, 3:7]
    object_quat_w = asset.data.root_quat_w

    active_quat = math_utils.quat_mul(goal_quat_w, math_utils.quat_conjugate(object_quat_w))
    # isaaclab quaternions are wxyz; the vector part is components [1:4]
    # (equivalent to active_quat[..., :3] under an xyzw convention).
    active_quat_vec = active_quat[..., 1:4]

    reward = torch.exp(-torch.norm(active_quat_vec, p=2, dim=-1) ** 2 * rot_scale / rot_temp)
    return reward


def track_position(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    pos_scale: float = 2.0,
    pos_temp: float = 0.5,
    max_pos_error: float | None = None,  # maximum position error to enable the reward
) -> torch.Tensor:
    """Reward for tracking the object position with an exponential (Gaussian) kernel.
    The reward is from https://arxiv.org/abs/2509.07445 Gemini-1.5-Flash.

    reward = exp(-||active_pos||_2^2 * pos_scale / pos_temp)

    where active_pos = goal_pos_e - object_pos_e is the position error in the env frame.
    If `max_pos_error` is provided, the reward is zeroed for envs whose position error
    exceeds the threshold.
    """
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)

    goal_pos_e = command_term.command[:, 0:3]
    object_pos_e = asset.data.root_pos_w - env.scene.env_origins
    active_pos = goal_pos_e - object_pos_e

    pos_error = torch.norm(active_pos, p=2, dim=-1)
    reward = torch.exp(-(pos_error ** 2) * pos_scale / pos_temp)

    if max_pos_error is not None:
        reward = torch.where(pos_error <= max_pos_error, reward, torch.zeros_like(reward))

    return reward
```

```python
    track_position = RewTerm(
        func=task_mdp.track_position,
        weight=2.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "object_pose",
            "pos_scale": 2.0,
            "pos_temp": 0.5,
        },
    )

    track_orientation = RewTerm(
        func=task_mdp.track_orientation,
        weight=5.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "object_pose",
            "rot_scale": 5.0,
            "rot_temp": 1.0,
        },
    )
```

```python
# THIS IS DIFFERENT FROM THE OFFICIAL ONE AS THE ORIENTATION IS RESAMPLED AROUND THE CURRENT OBJECT POSE, NOT THE DEFAULT POSE.
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
        # -- compute the number of consecutive successes
        successes = (
            self.metrics["orientation_error"] < self.cfg.orientation_success_threshold
        )
        self.metrics["consecutive_success"] += successes.float()

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # set position command from the current object pose at reset; stays fixed until next reset.
        idx: slice | Sequence[int] = slice(None) if env_ids is None else env_ids
        self.pos_command_w[idx] = self.object.data.root_pos_w[idx] + self.init_pos_offset
        self.pos_command_e[idx] = self.pos_command_w[idx] - self._env.scene.env_origins[idx]
        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        # position command is set in `reset()` and intentionally left unchanged on resample.

        # sample new orientation targets: fixed geodesic angle, uniform axis on S^2
        n = len(env_ids)
        angle = torch.full((n,), float(self.random_range), device=self.device)

        axis = torch.randn((n, 3), device=self.device)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        quat_delta = math_utils.quat_from_angle_axis(angle, axis)

        # apply delta to the current object orientation (set by load_grasp on reset)
        init_quat = self.object.data.root_quat_w[env_ids]
        quat = math_utils.quat_mul(init_quat, quat_delta)

        self.quat_command_w[env_ids] = (
            math_utils.quat_unique(quat) if self.cfg.make_quat_unique else quat
        )

    def _update_command(self):
        # update the command if goal is reached
        if self.cfg.update_goal_on_success:
            # compute the goal resets
            goal_resets = (
                self.metrics["orientation_error"]
                < self.cfg.orientation_success_threshold
            )
            goal_reset_ids = goal_resets.nonzero(as_tuple=False).squeeze(-1)
            # resample the goals
            self._resample(goal_reset_ids)

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
```

```python
@configclass
class InHandReOrientationCommandCfg(CommandTermCfg):
    """Configuration for the uniform 3D orientation command term.

    Please refer to the :class:`InHandReOrientationCommand` class for more details.
    """

    random_range: float = 1.0  # TODO: tune this one, and set up a curriculum for the range (e.g., 0.1->0.25->0.5->1.0)
    """Range for the random orientation."""

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

    update_goal_on_success: bool = MISSING
    """Whether to update the goal orientation when the goal orientation is reached."""

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
```