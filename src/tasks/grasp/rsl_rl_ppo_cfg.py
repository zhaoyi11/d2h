from isaaclab.utils import configclass

from src.tasks.common.rsl_rl_ppo_cfg_base import RslRlPpoCfgBase


@configclass
class LeapObjectRslRlPpoCfg(RslRlPpoCfgBase):
    experiment_name = "reorient"
    obs_groups = {"policy": ["policy", "perception"], "critic": ["policy", "perception"]}
    # obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    max_iterations = 1000
    save_interval = 250

