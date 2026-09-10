Use FlashSAC instead of RSL-RL to train D2H policies

     Context

     D2H (/home/yizhao/yi/D2H) currently trains dexterous manipulation policies with RSL-RL (PPO, on-policy). FlashSAC_dex (/home/yizhao/yi/FlashSAC_dex) is a compiled-kernel, off-policy SAC implementation that already ships an Isaac Lab adapter (flash_rl/envs.py:IsaacLabVectorEnv). Both projects target Isaac Lab 2.x on GPU, so the simulator
     layer is shared — only the training backend (runner, losses, networks, CLI) differs.

     The goal is to train D2H tasks with FlashSAC — starting with Reorient-v0 (closest analogue to FlashSAC's already-tested Isaac-Repose-Cube-* envs, so algorithm hyperparameters transfer most directly) — while preserving D2H's Hydra-driven env configs, grasp/object CLI overrides, and task registry. RSL-RL-based training remains available
     alongside FlashSAC (parallel trainer, no removals).

     Architectural approach

     Keep FlashSAC as a library (flash_rl package) and add a new trainer script inside D2H. Do not edit FlashSAC's algorithm code. Write one small adapter that bridges D2H's Hydra-built env into flash_rl's expected VectorEnv interface. This is cleaner than shimming D2H into FlashSAC's tyro/CLI world, because D2H tasks need Hydra + grasp CLI
     overrides that FlashSAC doesn't know about.

     D2H/scripts/flash_sac/train.py   <-- new entry (mirrors scripts/rsl_rl/train.py)
             │  1. AppLauncher + parse args (--task, --num_envs, --grasp_path, ...)
             │  2. @hydra_task_config builds env_cfg
             │  3. gym.make(task, cfg=env_cfg)  → Isaac Lab env
             │  4. wrap with FlashSACIsaacLabAdapter (new, small)
             │  5. build FlashSAC Config from agent_cfg_entry_point
             │  6. create_agent + training loop (copied/called from FlashSAC_dex/train.py)
             ▼
     flash_rl.agent.create_agent(...)  <-- unchanged
     flash_rl.IsaacLabVectorEnv API    <-- contract we satisfy

     Key reference points

     FlashSAC side (read-only, contract to satisfy):
     - FlashSAC_dex/train.py:74-160 — training loop (replay buffer fill → update → eval → ckpt)
     - FlashSAC_dex/flash_rl/envs.py:45-189 — IsaacLabVectorEnv contract: reset(), step(), observation_space, action_space, num_envs, max_episode_steps, and infos keys final_obs, actor_observation_size, asymmetric_obs, plus observations.critic in step-infos
     - FlashSAC_dex/flash_rl/config.py — Config, EnvConfig, AgentConfig, RunnerConfig dataclasses (tyro)
     - FlashSAC_dex/flash_rl/agent.py:create_agent — builds actor+critic+replay buffer from spaces + env_info
     - FlashSAC_dex/flash_rl/envs.py:16-29 — ACTION_BOUNDS dict; defaults to 1.0 for unknown envs

     D2H side (what must keep working):
     - D2H/scripts/rsl_rl/train.py:150-287 — reference pattern for CLI parsing, Hydra task config, env creation, grasp overrides
     - D2H/src/tasks/__init__.py — 8 registered tasks; each has a rsl_rl_cfg_entry_point
     - D2H/src/utils.py:apply_grasp_overrides_from_cli — must be applied to env_cfg before gym.make
     - D2H/src/tasks/common/rsl_rl_ppo_cfg_base.py — pattern for per-task RL config entry points; we mirror this for SAC
     - D2H env produces obs_dict["policy"] and may produce obs_dict["critic"] (asymmetric path — FlashSAC already handles this)

     Implementation steps

     1. Wire FlashSAC into D2H's uv environment

     Edit D2H/pyproject.toml to add flash_rl as an editable local dependency:

     [project]
     dependencies = [
         "flash_rl",
     ]

     [tool.uv.sources]
     flash_rl = { path = "../FlashSAC_dex", editable = true }

     Then in the D2H repo: uv sync (keeps Isaac Sim 5.1 / torch 2.9.1 pinned by D2H's install script; do not enable FlashSAC's isaaclab extra — D2H already provides Isaac Lab). Confirm python -c "import flash_rl; from flash_rl.agent import create_agent" works in D2H's .venv.

     2. Write the Isaac Lab → FlashSAC adapter

     New file D2H/src/flash_sac/env_adapter.py. Mirrors flash_rl.envs.IsaacLabVectorEnv, but takes an already-constructed gym env (since AppLauncher is already running and Hydra has built env_cfg). Copy the body of IsaacLabVectorEnv.__init__ (from self.num_envs = ... onward, i.e., skip the AppLauncher + parse_env_cfg block) and reset(), step(),
     close(), render() verbatim. Key behaviors to preserve:
     - populate single_observation_space / action_space from env.unwrapped.single_observation_space["policy"] (and critic if present)
     - infos["final_obs"] = concatenated next obs (FlashSAC uses this for terminal transitions)
     - infos["actor_observation_size"], infos["asymmetric_obs"], infos["time_outs"], infos["observations"]["critic"]
     - to_numpy=True conversion matches FlashSAC's train-loop expectations

     3. Add per-task FlashSAC config entry points

     New module D2H/src/tasks/common/flash_sac_cfg_base.py that produces a flash_rl.config.Config (or dict convertible to it). For each task that needs FlashSAC training, add e.g. src/tasks/pick_anyrotate/flash_sac_cfg.py exporting PickAnyRotateFlashSacCfg and register a new kwarg on the gym task:

     gym.register(id="Reorient-v0", ..., kwargs={
         ...,
         "rsl_rl_cfg_entry_point": "...",
         "flash_sac_cfg_entry_point": "src.tasks.reorient.flash_sac_cfg:LeapObjectFlashSacCfg",
     })

     Start with one task (Reorient-v0) to validate, then fan out. Reorient is single-group (policy obs only) in most D2H configs, so the symmetric-critic path is exercised first — avoiding asymmetric-obs complications until the core loop is verified.

     4. Create the new trainer script

     New file D2H/scripts/flash_sac/train.py. Structure (mirror scripts/rsl_rl/train.py):

     1. argparse with D2H's CLI args (--task, --num_envs, --seed, --grasp_path, --obj_urdf_path, --obj_scale, --max_iterations, --video, --distributed). Drop RSL-RL-specific flags; add FlashSAC-relevant ones (--num_interaction_steps, --updates_per_step, --buffer_size, --save_interval).
     2. AppLauncher(args_cli) — must happen before any Isaac imports, same pattern as RSL-RL trainer.
     3. @hydra_task_config(args_cli.task, "flash_sac_cfg_entry_point") → get env_cfg, agent_cfg.
     4. Apply _apply_grasp_overrides(env_cfg, args_cli), set env_cfg.scene.num_envs, env_cfg.seed, env_cfg.sim.device.
     5. env = gym.make(args_cli.task, cfg=env_cfg); handle MARL→single-agent and video wrapping as in RSL-RL script.
     6. train_env = FlashSACIsaacLabAdapter(env, device=agent_cfg.device, action_bounds=agent_cfg.action_bounds).
     7. Copy the training loop body from FlashSAC_dex/train.py:48-164 verbatim: create_agent(...), WandbTrainerLogger, replay-fill loop, agent.update(), eval/ckpt hooks. Drop the video recording path initially (D2H's gym.wrappers.RecordVideo already handles it).
     8. Write logs to D2H/logs/flash_sac/{experiment_name}/{timestamp}/ to parallel the RSL-RL layout; dump env/agent yaml via dump_yaml.

     5. Handle action bounds and observation groups

     - Most D2H tasks use RelativeJointPositionActionCfg with bounded outputs; FlashSAC's adapter clips actions to [-1, 1] * action_bounds. Expose action_bounds on each task's FlashSacCfg (default 1.0 — matches RSL-RL's clip_actions=1.0). Verify per task by checking env_cfg action scale.
     - If a D2H task's observations has only a policy group, the asymmetric path is skipped and FlashSAC trains a symmetric critic — no code change needed. If critic is defined, FlashSAC's existing asymmetric critic code path picks it up automatically.
     - D2H tasks with a perception group (e.g., point clouds in pick_screw) will need that group dropped or fused into policy/critic for SAC; flag as follow-up if encountered.

     6. Verification

     End-to-end smoke test on Reorient-v0 with small resource footprint:

     cd /home/yizhao/yi/D2H
     uv run python scripts/flash_sac/train.py \
         --task Reorient-v0 \
         --num_envs 256 \
         --num_interaction_steps 200 \
         --seed 42 \
         --headless

     Checks:
     - AppLauncher starts, env builds, observation_space.shape[-1] matches the RSL-RL-trained policy for the same task
     - FlashSAC prints buffer-fill progress, then enters update loop; loss numbers finite
     - W&B run logs episode_info/return, actor_loss, critic_loss
     - Checkpoint file appears under logs/flash_sac/pick_lift/…/stepN/

     Regression check: run the existing RSL-RL command (uv run python scripts/rsl_rl/train.py --task Reorient-v0 --num_envs 256 --max_iterations 5) to confirm nothing in pyproject.toml / src/tasks/__init__.py broke the original path.

     Full training run (after smoke test passes): Reorient-v0 with num_envs=4096, num_interaction_steps≈100_000, compare learning curve to the RSL-RL baseline on the same task, and to FlashSAC's published numbers on Isaac-Repose-Cube-Allegro-Direct-v0 (sanity check on order of magnitude).

     Files that will be created or modified

     Created:
     - D2H/scripts/flash_sac/train.py
     - D2H/src/flash_sac/__init__.py
     - D2H/src/flash_sac/env_adapter.py
     - D2H/src/tasks/common/flash_sac_cfg_base.py
     - D2H/src/tasks/reorient/flash_sac_cfg.py (first; add one per additional task as we fan out)

     Modified:
     - D2H/pyproject.toml (add flash_rl as editable path dep)
     - D2H/src/tasks/__init__.py (add flash_sac_cfg_entry_point kwargs)
     - D2H/uv.lock (regenerated by uv sync)

     Untouched:
     - Everything under FlashSAC_dex/flash_rl/*
     - D2H/scripts/rsl_rl/train.py and all rsl_rl_ppo_cfg.py files (RSL-RL path stays functional)

     Addendum: reward logging + eval cadence

     Context

     Smoke test passes but W&B shows no episodic return, only SAC losses and temperature. Two related issues:

     1. No reward in W&B. FlashSAC's loop (FlashSAC_dex/train.py:98-99, mirrored at scripts/flash_sac/train.py:247-248) gates reward logging on "episode_info" in env_infos. The D2H adapter (src/flash_sac/env_adapter.py:145-172) never populates that key, so the branch never fires. FlashSAC's own upstream IsaacLabVectorEnv.step() also never emits
     episode_info — upstream relies on its evaluate() pass for the return metric. We need to surface episodic return from D2H's Isaac Lab env directly.
     2. "Eval frequency" semantics. FlashSAC's upstream path calls evaluate(agent, eval_env, ...) every runner.evaluation_per_interaction_step, but Isaac Lab enforces one SimulationApp per process — there is no independent eval_env (upstream aliases train_env == eval_env, see flash_rl/envs.py:242), so a mid-training evaluate() would stomp the
     train env's state. Our trainer deliberately omits the eval call; the honest substitute is rolling train-time episodic return logged at the existing runner.logging_per_interaction_step cadence, which is already wired.

     Approach

     Populate infos["episode_info"] from the Isaac Lab extras dict inside the adapter.

     Isaac Lab's ManagerBasedRLEnv.step() returns an extras dict whose "log" sub-dict contains per-reset-batch mean Episode_Reward/{term} and Episode_Termination/{term} entries (already averaged by RewardManager.reset() over envs that reset on that step, normalized by max_episode_length_s). These are exactly the per-term reward curves D2H's
     RSL-RL runs show in W&B. Re-expose them under the key FlashSAC's loop already reads.

     Additionally, track per-env episodic return + length in the adapter so we can emit episode/return and episode/length (total, not per-term) whenever any env resets:

     - Maintain two torch.zeros(num_envs) buffers: _ep_return, _ep_length.
     - After each step: _ep_return += rew, _ep_length += 1.
     - For envs where terminations | truncations: push (_ep_return[i], _ep_length[i]) into a bounded FIFO (e.g. collections.deque(maxlen=num_envs * 4)), then zero those entries.
     - On each step where any env finished, emit:
     infos["episode_info"] = {
         "episode/return": float(np.mean(finished_returns)),
         "episode/length": float(np.mean(finished_lengths)),
         **{f"Episode_Reward/{k}": float(v) for k, v in extras.get("log", {}).items() if k.startswith("Episode_Reward/")},
     }
     - If no env finished this step, do not set episode_info (matches FlashSAC's "only log when present" semantics; WandbTrainerLogger accumulates across a logging window anyway).

     For Reorient, optionally also surface env.command_manager.get_term("object_pose").metrics["consecutive_success"].mean() under episode/success — this is the task-level metric D2H cares about. Guard behind a hasattr check so the adapter stays task-agnostic.

     Reward-pipeline facts to rely on

     - env.unwrapped.reward_manager._episode_sums[term_name] — per-env running sum, shape (num_envs,); reset to 0 per-env inside RewardManager.reset() on termination.
     - env.unwrapped.episode_length_buf — per-env step count, shape (num_envs,).
     - extras["log"]["Episode_Reward/{term}"] — mean across the reset batch, scalar (already a Python float after _recursive_to_numpy). Only present on steps where at least one env reset. These will be nan when no env resets on a given step — we must filter those out before emitting.
     - The current adapter step() discards extras (line 152, underscore binding). That is the single-line change at the adapter level.

     Code changes

     - D2H/src/flash_sac/env_adapter.py
       - Capture extras in step() (replace _ at line 152).
       - Initialize _ep_return, _ep_length, and _ep_fifo in __init__.
       - Update buffers, collect finished episodes, build episode_info dict, attach to infos. Convert to numpy under self.to_numpy.
     - D2H/scripts/flash_sac/train.py
       - Already reads episode_info; no change needed to the loop itself.
       - Add an explicit note in the top docstring that "evaluation" under Isaac Lab means rolling train-time return logged at logging_per_interaction_step cadence — there is no separate eval env.
       - Optionally expose --logging_interval CLI flag that maps to flash_sac_cfg.runner.logging_per_interaction_step (currently only tunable via the task's flash_sac_cfg.py).

     Nothing changes under flash_rl/* or in the base FlashSacCfgBase; the fix is isolated to the adapter and an optional CLI knob.

     Verification

     1. Re-run the smoke test. uv run python scripts/flash_sac/train.py --task Reorient-v0 --num_envs 256 --num_interaction_steps 500 --headless. W&B run should now show episode/return, episode/length, and Episode_Reward/* curves alongside the existing actor_loss, critic_loss, alpha.
     2. Cross-check magnitudes against RSL-RL. For the same task and similar num_envs, early episode/return values from FlashSAC should be in the same order of magnitude as the Episode_Reward/… values D2H's RSL-RL run logs (both read the same reward terms from RewardManager). Large mismatch → buffer-bookkeeping bug.
     3. Finished-episode count sanity. Reorient's max_episode_length is O(hundreds) of steps; at num_envs=256, expect several finished episodes per logging_per_interaction_step=1000 window. If episode/return is constant or missing, the FIFO is empty — verify _ep_return / _ep_length zeroing.

     Files modified

     - D2H/src/flash_sac/env_adapter.py (main fix)
     - D2H/scripts/flash_sac/train.py (docstring + optional CLI flag)

● Read(/home/yizhao/yi/D2H/src/flash_sac/env_adapter.py)
  ⎿  Unchanged since last read                                                            
                                                             
● Update(/home/yizhao/yi/D2H/src/flash_sac/env_adapter.py)                                                                                                                                                                                                                                                                                                  
  ⎿  Added 2 lines                                                                                                                                                                                                                                                                                                                                          
       9                                                                                                                                                                                                                                                                                                                                                    
      10  from __future__ import annotations                                                                                                                                                                                                                                                                                                                
      11                                                                                                                                                                                                                                                                                                                                                    
      12 +import math                                                                                                                                                                                                                                                                                                                                       
      13 +from collections import deque                                                                                                                                                                                                                                                                                                                     
      14  from typing import Any, Sequence, Union, cast                                                                                                                                                                                                                                                                                                     
      15                                                                                                                                                                                                                                                                                                                                                    
      16  import gymnasium as gym                                                                                                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                                                                                            
● Now initialize tracking buffers in __init__.                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                            
● Update(/home/yizhao/yi/D2H/src/flash_sac/env_adapter.py)                                                                                                                                                                                                                                                                                                  
  ⎿  Added 11 lines                                                                                                                                                                                                                                                                                                                                         
      114          )                                                                                                                                                                                                                                                                                                                                        
      115          self.action_space = batch_space(self.single_action_space, self.num_envs)                                                                                                                                                                                                                                                                 
      116                                                                                                                                                                                                                                                                                                                                                   
      117 +        # Per-env episodic return / length trackers. FlashSAC's loop gates wandb                                                                                                                                                                                                                                                                 
      118 +        # reward logging on infos["episode_info"] being present; upstream's                                                                                                                                                                                                                                                                      
      119 +        # IsaacLabVectorEnv relies on evaluate() to produce returns, but that                                                                                                                                                                                                                                                                    
      120 +        # path is unavailable under Isaac Lab's single-SimulationApp constraint.                                                                                                                                                                                                                                                               
      121 +        # We compute rolling train-time returns here instead.                                                                                                                                                                                                                                                                                    
      122 +        self._ep_return = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)                                                                                                                                                                                                                                                  
      123 +        self._ep_length = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)                                                                                                                                                                                                                                                       
      124 +        fifo_cap = max(self.num_envs * 4, 64)                                                                                                                                                                                                                                                                                                  
      125 +        self._ep_return_fifo: deque[float] = deque(maxlen=fifo_cap)                                                                                                                                                                                                                                                                            
      126 +        self._ep_length_fifo: deque[int] = deque(maxlen=fifo_cap)                                                                                                                                                                                                                                                                              
      127 +                                                                                                                                                                                                                                                                                                                                                 
      128      def _build_obs(self, obs_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:                                                                                                                                                                                                                                         
      129          actor_obs = _concat_groups(obs_dict, self._actor_groups)                                                                                                                                                                                                                                                                                 
      130          if self.asymmetric_obs:                                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                            
● Zero the running buffers on reset() (keep the FIFO so a warm start still has a return to log).                                                                                                                                                                                                                                                            
                                       
● Update(/home/yizhao/yi/D2H/src/flash_sac/env_adapter.py)    
  ⎿  Added 3 lines                                   
      149                  unwrapped.episode_length_buf, high=int(self.max_episode_steps)
      150              )                             
      151                                              
      152 +        self._ep_return.zero_()                                                                                                                                                                                                                                                                                                                  
      153 +        self._ep_length.zero_()                                                                                                                                                                                                                                                                                                           
      154 +                                                                                                                                                                                                                                                                                                                                                 
      155          if self.to_numpy:                                                                                                                                                                                                                                                                                                                      
      156              obs = obs.cpu().numpy()                                                                                                                                                                                                                                                                                                              
      157              infos = _recursive_to_numpy(infos)                                                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                                                                            
● Now the main step() change — capture extras, update trackers, and emit episode_info.                                                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                                                                                                          
● Update(/home/yizhao/yi/D2H/src/flash_sac/env_adapter.py)                                                                                                                                                                                                                                                                                                
  ⎿  Added 57 lines, removed 1 line                                                                                                                                                                                                                                                                                                                         
      165              torch_actions = torch.from_numpy(np.asarray(actions)).to(self.device)                                                                                                                                                                                                                                                                
      166          torch_actions = torch.clamp(torch_actions, -1.0, 1.0) * self.action_bounds                                                                                                                                                                                                                                                               
      167                                                                                                                                                                                                                                                                                                                                                   
      168 -        obs_dict, rew, terminations, truncations, _ = cast(Any, self._env.step(torch_actions))                                                                                                                                                                                                                                                   
      168 +        obs_dict, rew, terminations, truncations, extras = cast(Any, self._env.step(torch_actions))                                                                                                                                                                                                                                              
      169          obs, critic_obs = self._build_obs(obs_dict)                                                                                                                                                                                                                                                                                              
      170                                                                                                                                                                                                                                                                                                                                                 
      171          # NOTE: IsaacLab auto-resets terminated envs inside step(), so obs here is                                                                                                                                                                                                                                                               
     ...                                                                                                                                                                                                                                                                                                                                                  
      179              "final_obs": obs,                                                                                                                                                                                                                                                                                                                    
      180          }                                                                                                                                                                                                                                                                                                                                        
      181                                                                                                                                                                                                                                                                                                                                                 
      182 +        self._ep_return += rew                                                                                                                                                                                                                                                                                                                 
      183 +        self._ep_length += 1                                                                                                                                                                                                                                                                                                                     
      184 +        done_mask = terminations | truncations                                                                                                                                                                                                                                                                                                   
      185 +        episode_info = self._build_episode_info(done_mask, extras)                                                                                                                                                                                                                                                                               
      186 +        if episode_info:                                                                                                                                                                                                                                                                                                                         
      187 +            infos["episode_info"] = episode_info                                                                                                                                                                                                                                                                                                 
      188 +                                                                                                                                                                                                                                                                                                                                                 
      189          if self.to_numpy:
      190              obs = obs.cpu().numpy()
      191              rew = rew.cpu().numpy()
     ...
      194              infos = _recursive_to_numpy(infos)
      195          return obs, rew, terminations, truncations, infos
      196  
      197 +    def _build_episode_info(self, done_mask: torch.Tensor, extras: dict[str, Any]) -> dict[str, float]:                                                                                                                                                                                                                                   
      198 +        """Produce the dict FlashSAC's train loop logs under ``infos["episode_info"]``.                                                                                                                                                                                                                                                   
      199 +                                                                                                                                                                                                                                                                                                                                          
      200 +        Combines two sources:                                                                                                                                                                                                                                                                                                             
      201 +        - Per-env running return / length we track here → ``episode/return``,                                                                                                                                                                                                                                                             
      202 +          ``episode/length``. Only emitted on steps where at least one env reset,                                                                                                                                                                                                                                                         
      203 +          since those are the only steps that produce fresh samples.                                                                                                                                                                                                                                                                      
      204 +        - IsaacLab's ``extras["log"]`` — per-reset-batch mean of                                                                                                                                                                                                                                                                          
      205 +          ``Episode_Reward/{term}`` and ``Episode_Termination/{term}`` entries,                                                                                                                                                                                                                                                           
      206 +          populated by ``RewardManager.reset()``. NaN entries (no reset this                                                                                                                                                                                                                                                              
      207 +          step) are filtered out.                                                                                                                                                                                                                                                                                                         
      208 +        """                                                                                                                                                                                                                                                                                                                               
      209 +        info: dict[str, float] = {}                                                                                                                                                                                                                                                                                                       
      210 +                                                                                                                                                                                                                                                                                                                                          
      211 +        if bool(done_mask.any()):                                                                                                                                                                                                                                                                                                         
      212 +            finished_returns = self._ep_return[done_mask].detach().cpu().tolist()                                                                                                                                                                                                                                                         
      213 +            finished_lengths = self._ep_length[done_mask].detach().cpu().tolist()                                                                                                                                                                                                                                                         
      214 +            self._ep_return_fifo.extend(finished_returns)                                                                                                                                                                                                                                                                                 
      215 +            self._ep_length_fifo.extend(int(x) for x in finished_lengths)                                                                                                                                                                                                                                                                 
      216 +            self._ep_return[done_mask] = 0.0                                                                                                                                                                                                                                                                                              
      217 +            self._ep_length[done_mask] = 0                                                                                                                                                                                                                                                                                                
      218 +                                                                                                                                                                                                                                                                                                                                          
      219 +        if self._ep_return_fifo:                                                                                                                                                                                                                                                                                                          
      220 +            info["episode/return"] = float(np.mean(self._ep_return_fifo))                                                                                                                                                                                                                                                                 
      221 +            info["episode/length"] = float(np.mean(self._ep_length_fifo))                                                                                                                                                                                                                                                                 
      222 +                                                                                                                                                                                                                                                                                                                                          
      223 +        log = extras.get("log") if isinstance(extras, dict) else None                                                                                                                                                                                                                                                                     
      224 +        if isinstance(log, dict):                                                                                                                                                                                                                                                                                                         
      225 +            for key, value in log.items():                                                                                                                                                                                                                                                                                                
      226 +                if not (key.startswith("Episode_Reward/") or key.startswith("Episode_Termination/")):                                                                                                                                                                                                                                     
      227 +                    continue                                                                                                                                                                                                                                                                                                              
      228 +                scalar = value.item() if isinstance(value, torch.Tensor) else float(value)                                                                                                                                                                                                                                                
      229 +                if math.isnan(scalar):                                                                                                                                                                                                                                                                                                    
      230 +                    continue                                                                                                                                                                                                                                                                                                              
      231 +                info[key] = scalar                                                                                                                                                                                                                                                                                                        
      232 +                                                                                                                                                                                                                                                                                                                                          
      233 +        unwrapped: Any = self._env.unwrapped                                                                                                                                                                                                                                                                                              
      234 +        cmd_mgr = getattr(unwrapped, "command_manager", None)                                                                                                                                                                                                                                                                             
      235 +        if cmd_mgr is not None:                                                                                                                                                                                                                                                                                                           
      236 +            try:                                                                                                                                                                                                                                                                                                                          
      237 +                term = cmd_mgr.get_term("object_pose")                                                                                                                                                                                                                                                                                    
      238 +                success = term.metrics.get("consecutive_success")                                                                                                                                                                                                                                                                         
      239 +                if isinstance(success, torch.Tensor):                                                                                                                                                                                                                                                                                     
      240 +                    info["episode/consecutive_success"] = float(success.float().mean().item())                                                                                                                                                                                                                                            
      241 +            except (KeyError, AttributeError, ValueError):                                                                                                                                                                                                                                                                                
      242 +                pass                                                                                                                                                                                                                                                                                                                      
      243 +                                                                                                                                                                                                                                                                                                                                          
      244 +        return info                                                                                                                                                                                                                                                                                                                       
      245 +                                                                                                                                                                                                                                                                                                                                          
      246      def close(self, **kwargs: Any) -> None:
      247          # Leave the underlying env + simulation_app alive; the trainer script owns
      248          # lifecycle (matches FlashSAC's upstream IsaacLabVectorEnv.close no-op).

● Now update the trainer script: docstring clarification + --logging_interval CLI flag.
                                       
● Update(/home/yizhao/yi/D2H/scripts/flash_sac/train.py)       
  ⎿  Added 12 lines, removed 3 lines                 
       3  Mirrors the layout of ``scripts/rsl_rl/train.py``: AppLauncher is launched from
       4  CLI args before any Isaac import, then we build the env via ``gym.make`` using a
       5  Hydra-resolved ``env_cfg``, wrap it in ``FlashSACIsaacLabAdapter``, and drive
       6 -FlashSAC's training loop (replay buffer fill → SAC update → eval → checkpoint).                                                                                                                                                                                                                                                                   
       7 -The loop body is adapted from ``FlashSAC_dex/train.py``; we keep the same                                                                                                                                                                                                                                                                  
       8 -semantics but source hyperparameters from each task's ``flash_sac_cfg_entry_point``.                                                                                                                                                                                                                                                              
       6 +FlashSAC's training loop (replay buffer fill → SAC update → checkpoint). The                                                                                                                                                                                                                                                                    
       7 +loop body is adapted from ``FlashSAC_dex/train.py``; we keep the same semantics                                                                                                                                                                                                                                                                   
       8 +but source hyperparameters from each task's ``flash_sac_cfg_entry_point``.                                                                                                                                                                                                                                                                      
       9 +                                                                                                                                                                                                                                                                                                                                                
      10 +Note on "evaluation frequency": FlashSAC's upstream loop calls ``evaluate()``                                                                                                                                                                                                                                                                   
      11 +on a separate ``eval_env`` every ``runner.evaluation_per_interaction_step``.                                                                                                                                                                                                                                                               
      12 +Isaac Lab enforces a single ``SimulationApp`` per process, so we cannot stand                                                                                                                                                                                                                                                              
      13 +up an independent eval env mid-training. Instead, the adapter populates                                                                                                                                                                                                                                                                    
      14 +``infos["episode_info"]`` with rolling train-time episodic return + length                                                                                                                                                                                                                                                                 
      15 +(plus per-term ``Episode_Reward/*`` from IsaacLab's reward manager), and the                                                                                                                                                                                                                                                               
      16 +trainer flushes those to W&B every ``runner.logging_per_interaction_step``                                                                                                                                                                                                                                                                 
      17 +steps. Use ``--logging_interval`` to tune that cadence.                                                                                                                                                                                                                                                                                    
      18  """
      19  
      20  """Launch Isaac Sim Simulator first."""

● Update(/home/yizhao/yi/D2H/scripts/flash_sac/train.py)       
  ⎿  Added 1 line                      
      45  parser.add_argument("--buffer_min_length", type=int, default=None, help="Buffer-fill size before updates start.")
      46  parser.add_argument("--buffer_max_length", type=int, default=None, help="Replay buffer capacity.")
      47  parser.add_argument("--save_interval", type=int, default=None, help="Save checkpoint every N interaction steps.")
      48 +parser.add_argument("--logging_interval", type=int, default=None, help="Flush metrics to W&B every N interaction steps.")                                                                                                                                                                                                                  
      49  parser.add_argument("--no_wandb", action="store_true", default=False, help="Disable Weights & Biases logging.")
      50  parser.add_argument("--no_compile", action="store_true", default=False, help="Disable torch.compile in FlashSAC networks.")                                                                                                                                                                                                                       
      51  parser.add_argument("--no_amp", action="store_true", default=False, help="Disable mixed-precision training.")
                                                                                                                                                                                                                                                                                                                                                            
● Update(/home/yizhao/yi/D2H/scripts/flash_sac/train.py)                                                                                                                                                                                                                                                                                                  
  ⎿  Added 2 lines                                                                                                                                                                                                                                                                                                                                          
      118          flash_sac_cfg.agent.buffer_max_length = args.buffer_max_length                                                                                                                                                                                                                                                                           
      119      if args.save_interval is not None:                                                                                                                                                                                                                                                                                                         
      120          flash_sac_cfg.runner.save_checkpoint_per_interaction_step = args.save_interval
      121 +    if args.logging_interval is not None:                                                                                                                                                                                                                                                                                                 
      122 +        flash_sac_cfg.runner.logging_per_interaction_step = args.logging_interval                                                                                                                                                                                                                                                         
      123      if args.device is not None:                                                                                                                                                                                                                                                                                                                  
      124          flash_sac_cfg.device = args.device
      125          flash_sac_cfg.agent.device_type = args.device                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                          
● Verify the final adapter is coherent.                                                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                                                                          
● Read(/home/yizhao/yi/D2H/src/flash_sac/env_adapter.py)                                                                                                                                                                                                                                                                                                  
  ⎿  Read 253 lines                                  
                                                                  
● Quickly verify WandbTrainerLogger.update_metric accepts arbitrary string keys (not just identifiers), since our keys contain /.
                                                       
● Search(pattern: "def update_metric|class WandbTrainerLogger", path: "/home/yizhao/yi/FlashSAC_dex/flash_rl")                                                                                                                                                                                                                                              
  ⎿  Found 3 lines                                   
     flash_rl/utils.py:79:class WandbTrainerLogger:                                                                                                                                                                                                                                                                                                         
     flash_rl/utils.py:94:    def update_metric(self, **kwargs: Any) -> None:                                                                                                                                                                                                                                                                             
     flash_rl/utils.py:127:    def update_metric(self, **kwargs: Any) -> None:                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                            
● Read(/home/yizhao/yi/FlashSAC_dex/flash_rl/utils.py · lines 79-138)                                                                                                                                                                                                                                                                                       
  ⎿  Read 60 lines                                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                          
● Done. Summary of changes:                                                                                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                                                                            
  src/flash_sac/env_adapter.py                                                                                                                                                                                                                                                                                                                              
  - __init__: allocate _ep_return, _ep_length per-env buffers and bounded FIFOs for finished episodes (cap = max(num_envs * 4, 64)).                                                                                                                                                                                                                      
  - reset(): zero the per-env buffers (keep the FIFO so a fresh log window always has a return).                                                                                                                                                                                                                                                            
  - step(): capture extras (was discarded), accumulate return/length, snapshot + zero finished envs, build infos["episode_info"].
  - New _build_episode_info(): emits episode/return, episode/length, all Episode_Reward/* + Episode_Termination/* from extras["log"] (NaN-filtered), plus episode/consecutive_success for Reorient (guarded by try/except so it's task-agnostic).                                                                                                           
                                                                                                                                                                                                                                                                                                                                                            
  scripts/flash_sac/train.py                                                                                                                                                                                                                                                                                                                                
  - Docstring explains why there's no evaluate() call (Isaac Lab's one-SimulationApp constraint) and that train-time rolling return substitutes.                                                                                                                                                                                                            
  - New --logging_interval CLI flag → runner.logging_per_interaction_step.                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                            
  FlashSAC's existing if "episode_info" in env_infos branch now fires, and WandbTrainerLogger.update_metric averages the values across each logging window. Slash-containing keys (Episode_Reward/…) unpack fine through **kwargs.                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                            
  Smoke test it with:                                                                                                                                                                                                                                                                                                                                       
  uv run python scripts/flash_sac/train.py --task Reorient-v0 --num_envs 256 --num_interaction_steps 500 --logging_interval 50 --headless                                                                                                                                                                                                                   
                                      