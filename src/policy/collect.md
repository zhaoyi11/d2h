  python -m src.policy.data_collection \
      --task Reorient_Play-v0 --num_envs 4 --num_episodes 1024 \
      --checkpoint <path/to/model.pt> --headless --seed 0 --action_noise_std_max 0.1
      <!-- --action_noise_std_max 0.5 -->
