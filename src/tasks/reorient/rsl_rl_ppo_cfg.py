from isaaclab.utils import configclass

from src.tasks.common.rsl_rl_ppo_cfg_base import RslRlPpoCfgBase, RslRlPpoAlgorithmCfg


@configclass
class LeapObjectRslRlPpoCfg(RslRlPpoCfgBase):
    experiment_name = "reorient"
    obs_groups = {"policy": ["policy", "perception"], "critic": ["policy", "perception"]}
    max_iterations = 15000
    save_interval = 250
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # entropy_coef=0.005,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )