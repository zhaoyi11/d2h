"""Rotate-object HRL task driven by the distilled hand student."""

from isaaclab.utils import configclass

from src.tasks.rotate_knob_distill.env_cfg import (
    ObservationsCfg as RotateKnobDistillObservationsCfg,
)
from src.tasks.rotate_object.env_cfg import (
    DexsuiteFrankaLeapRotateObjectHrlEnvCfg,
)


@configclass
class DexsuiteFrankaLeapRotateObjectStudentHrlEnvCfg(
    DexsuiteFrankaLeapRotateObjectHrlEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.observations.low_level = RotateKnobDistillObservationsCfg.StudentCfg()


__all__ = ["DexsuiteFrankaLeapRotateObjectStudentHrlEnvCfg"]
