from isaaclab.utils import configclass

from src.tasks.rotate_object.rsl_rl_ppo_cfg import RotateObjectRslRlPpoCfg


@configclass
class RotateObjectStudentRslRlPpoCfg(RotateObjectRslRlPpoCfg):
    experiment_name = "rotate_object_student"


__all__ = ["RotateObjectStudentRslRlPpoCfg"]
