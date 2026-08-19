from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from src.tasks.common.rsl_rl_ppo_cfg_base import RslRlPpoCfgBase


@configclass
class UnscrewOmniResetRslRlPpoCfg(RslRlPpoCfgBase):
    experiment_name = "unscrew_omnireset"
    obs_groups = {
        "policy": ["policy", "proprio"],
        "critic": ["policy", "proprio"],
    }
    max_iterations = 15000
    save_interval = 250
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
