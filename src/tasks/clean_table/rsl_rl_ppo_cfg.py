from isaaclab.utils import configclass

from src.tasks.common.rsl_rl_ppo_cfg_base import RslRlPpoCfgBase


@configclass
class CleanTableRslRlPpoCfg(RslRlPpoCfgBase):
    experiment_name = "clean_table"
    obs_groups = {"policy": ["policy", "proprio", "perception"], "critic": ["policy", "proprio", "perception"]}
    max_iterations = 15000
    save_interval = 250
