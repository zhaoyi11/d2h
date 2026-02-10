from isaaclab.managers.action_manager import ActionTerm
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.utils import configclass

from .actions import EMACumulativeRelativeJointPositionAction


@configclass
class EMACumulativeRelativeJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the binary joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = EMACumulativeRelativeJointPositionAction
    """Class type."""

    alpha: float | dict[str, float] = 1.0
    """The weight for the moving average (float or dict of regex expressions). Defaults to 1.0.

    If set to 1.0, the processed action is applied directly without any moving average window.
    """

    joint_lower_limit: list[float] | None = None
    joint_upper_limit: list[float] | None = None
    """The lower and upper limits for the joint positions."""

    def __post_init__(self):
        super_post_init = getattr(super(), "__post_init__", None)
        if callable(super_post_init):
            super_post_init()

        has_lower = self.joint_lower_limit is not None
        has_upper = self.joint_upper_limit is not None
        if has_lower != has_upper:
            raise ValueError(
                "joint_lower_limit and joint_upper_limit must either both be set or both be None."
            )
        if has_lower and len(self.joint_lower_limit) != len(self.joint_upper_limit):
            raise ValueError(
                "joint_lower_limit and joint_upper_limit must have the same length."
            )
