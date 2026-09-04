"""Behavior-cloning evaluation config for recorded knob rotation."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass

import src.tasks.common.mdps as task_mdps
from src.tasks.rotate_object_once.env_cfg import ObservationsCfg as RotateObjectObservationsCfg
from src.tasks.rotate_object_omnireset.env_cfg import (
    DexsuiteFrankaLeapRotateObjectOmniResetEnvCfg,
    ObservationsCfg as OmniResetObservationsCfg,
)


def _hand_base_command(env, command_name: str):
    return env.command_manager.get_command(command_name)[:, 7:14]


@configclass
class ObservationsCfg(OmniResetObservationsCfg):
    @configclass
    class PolicyCfg(RotateObjectObservationsCfg.LowLevelObsCfg):
        arm_joint_pos = ObsTerm(
            func=task_mdps.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
            },
        )
        arm_joint_vel = ObsTerm(
            func=task_mdps.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
            },
        )
        hand_base_command = ObsTerm(
            func=_hand_base_command,
            params={"command_name": "object_pose"},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    arm_action = task_mdps.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=1.0,
    )
    hand_action = task_mdps.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["a_.*"],
        alpha=0.5,
        rescale_to_limits=True,
    )


@configclass
class DexsuiteFrankaLeapRotateObjectBCEnvCfg(
    DexsuiteFrankaLeapRotateObjectOmniResetEnvCfg
):
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    curriculum: None = None

    def __post_init__(self):
        super().__post_init__()
        self.scene.fingertip_transforms = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Franka_LeapHand/leap_hand_right/base",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=(
                        "{ENV_REGEX_NS}/Robot/Franka_LeapHand/leap_hand_right/"
                        "thumb_fingertip"
                    ),
                    offset=OffsetCfg(pos=(0.0, -0.045, -0.015)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path=(
                        "{ENV_REGEX_NS}/Robot/Franka_LeapHand/leap_hand_right/"
                        "fingertip"
                    ),
                    offset=OffsetCfg(pos=(0.0, -0.03, 0.015)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path=(
                        "{ENV_REGEX_NS}/Robot/Franka_LeapHand/leap_hand_right/"
                        "fingertip_2"
                    ),
                    offset=OffsetCfg(pos=(0.0, -0.03, 0.015)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path=(
                        "{ENV_REGEX_NS}/Robot/Franka_LeapHand/leap_hand_right/"
                        "fingertip_3"
                    ),
                    offset=OffsetCfg(pos=(0.0, -0.03, 0.015)),
                ),
            ],
            debug_vis=False,
        )
        self.observations.proprio = None
        self.commands.object_pose.include_hand_base_command = True
        # TODO: Switch to the full reset-state pool when full-goal evaluation is desired.
        self.events.reset_from_dataset.params["hardest_only"] = True


__all__ = ["DexsuiteFrankaLeapRotateObjectBCEnvCfg"]
