from dataclasses import MISSING

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

    joint_lower_limit: list[float] = None
    joint_upper_limit: list[float] = None
    """The lower and upper limits for the joint positions."""
    