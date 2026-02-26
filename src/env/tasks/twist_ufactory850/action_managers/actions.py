from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.actions import JointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from . import actions_cfg


class EMACumulativeRelativeJointPositionAction(JointPositionAction):
    cfg: actions_cfg.EMACumulativeRelativeJointPositionActionCfg
    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: actions_cfg.EMACumulativeRelativeJointPositionActionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # parse and save the moving average weight
        if isinstance(cfg.alpha, float):
            # check that the weight is in the valid range
            if not 0.0 <= cfg.alpha <= 1.0:
                raise ValueError(f"Moving average weight must be in the range [1, 1]. Got {cfg.alpha}.")
            self._alpha = cfg.alpha
        elif isinstance(cfg.alpha, dict):
            self._alpha = torch.ones((env.num_envs, self.action_dim), device=self.device)
            # resolve the dictionary config
            index_list, names_list, value_list = string_utils.resolve_matching_names_values(
                cfg.alpha, self._joint_names
            )
            # check that the weights are in the valid range
            for name, value in zip(names_list, value_list):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"Moving average weight must be in the range [0, 1]. Got {value} for joint {name}."
                    )
            self._alpha[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(
                f"Unsupported moving average weight type: {type(cfg.alpha)}. Supported types are float and dict."
            )

        # initialize the previous targets
        self._prev_applied_actions = torch.zeros_like(self.processed_actions)
        # initialize the cumulative del action
        self.del_action = torch.zeros((self._env.num_envs, self.action_dim), device=self._env.device)
        self.init_joint_pos = self._asset.data.joint_pos[:, self._joint_ids].clone()
        self.joint_lower_limit = torch.tensor(cfg.joint_lower_limit, device=self.device) if cfg.joint_lower_limit is not None else None
        self.joint_upper_limit = torch.tensor(cfg.joint_upper_limit, device=self.device) if cfg.joint_upper_limit is not None else None
    
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # check if specific environment ids are provided
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        # reset history to current joint positions
        self._prev_applied_actions[env_ids, :] = self._asset.data.joint_pos[env_ids][:, self._joint_ids].clone()
        # reset the del action
        self.del_action[env_ids, :] = torch.zeros((env_ids.shape[0], self.action_dim), device=self.device)
        self.init_joint_pos[env_ids, :] = self._asset.data.joint_pos[env_ids][:, self._joint_ids].clone()

    def process_actions(self, actions: torch.Tensor):
        # apply affine transformations
        super().process_actions(actions)
        # compute the del action
        self._processed_actions += self.del_action
        self.del_action = self._processed_actions.clone()
        # add the initial position
        self._processed_actions += self.init_joint_pos.clone()
        # set position targets as moving average
        ema_actions = self._alpha * self._processed_actions
        ema_actions += (1.0 - self._alpha) * self._prev_applied_actions
        # clamp the targets
        if self.joint_lower_limit is not None and self.joint_upper_limit is not None:
            self._processed_actions[:] = torch.clamp(
                ema_actions,
                self.joint_lower_limit,
                self.joint_upper_limit,
            )
        else:
            self._processed_actions[:] = ema_actions
        # update previous targets
        self._prev_applied_actions[:] = self._processed_actions[:]


class EMACumulativeRelativeJointPositionActionEval:
    cfg: actions_cfg.EMACumulativeRelativeJointPositionActionCfg
    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(self, action_dim, init_joint_pos, alpha, device, lower_limit, upper_limit) -> None:
        # parse and save the moving average weight
        if isinstance(alpha, float):
            # check that the weight is in the valid range
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"Moving average weight must be in the range [0, 1]. Got {alpha}.")
            self._alpha = alpha
        else:
            raise ValueError(
                f"Unsupported moving average weight type: {type(alpha)}. Supported types are float and dict."
            )
        self.action_dim = action_dim
        # initialize the previous targets
        self._prev_applied_actions = init_joint_pos.clone()
        # initialize the cumulative del action
        self.del_action = torch.zeros(self.action_dim, device=device)
        self.init_joint_pos = init_joint_pos.clone()
        self.processed_actions = torch.zeros(self.action_dim, device=device)
        self.device = device
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
    
    def reset(self, init_joint_pos) -> None:
        # reset history to current joint positions
        self._prev_applied_actions = init_joint_pos.clone()
        # reset the del action
        self.del_action = torch.zeros(self.action_dim, device=self.device)
        self.init_joint_pos = init_joint_pos.clone()

    def process_actions(self, actions: torch.Tensor):
        # apply affine transformations
        self.processed_actions = actions
        # compute the del action
        self.processed_actions += self.del_action
        self.del_action = self.processed_actions.clone()
        # add the initial position
        self.processed_actions += self.init_joint_pos.clone()
        # set position targets as moving average
        ema_actions = self._alpha * self.processed_actions
        ema_actions += (1.0 - self._alpha) * self._prev_applied_actions
        # clamp the targets
        self.processed_actions[:] = torch.clamp(
            ema_actions,
            self.lower_limit,
            self.upper_limit,
        )
        # update previous targets
        self._prev_applied_actions[:] = self.processed_actions[:]
        return self.processed_actions