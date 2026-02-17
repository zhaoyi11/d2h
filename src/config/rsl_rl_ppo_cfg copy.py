from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
)


@configclass
class AllegroCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    clip_actions = 1.0
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = "allegro_cube"
    obs_groups = {
        # Include proprioception to improve grasp approach/control observability.
        "policy": ["policy", "proprio"],
        "critic": ["policy", "proprio"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=0.001,
        schedule="adaptive",
        gamma=0.998,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # rnd_cfg=RslRlRndCfg(
        #     weight=10.0,
        #     reward_normalization=False,
        #     state_normalization=True,
        #     learning_rate=0.001,
        #     predictor_hidden_dims=[32, 32],
        #     target_hidden_dims=[32],
        #     # weight_schedule=RslRlRndCfg.LinearWeightScheduleCfg(
        #     #     final_value=0.1,
        #     #     initial_step=100,
        #     #     final_step=3000,
        #     # ),
        # ),
    )


@configclass
class AllegroCubeNoVelObsPPORunnerCfg(AllegroCubePPORunnerCfg):
    experiment_name = "allegro_cube_no_vel_obs"
