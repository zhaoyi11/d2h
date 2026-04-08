To generate grasps
```sh
python3 scripts/rsl_rl/train.py --task Grasp-v0 --num_envs 8192 --headless --track --logger wandb --wandb_entity aria-ic
```

To collect grasp
```sh
python3 scripts/rsl_rl/play.py --task GraspCollect-v0 --num_envs 8192 --headless --checkpoint /home/yizhao/yi/D2H/logs/rsl_rl/reorient/2026-03-31_15-46-39/model_1250.pt
```
This will save the generated grasps to `grasp_data.npy`

To train the reorient policy
```sh
python3 scripts/rsl_rl/train.py --task Reorient-v0 --grasp_path /home/yizhao/yi/D2H/grasp_data.npy --num_envs=8192 --headless --track --logger wandb --wandb_entity aria-ic
```


To eval the reorient policy
```sh
 python3 scripts/rsl_rl/play.py --task Reorient-v0 --num_envs 8 --checkpoint /home/yizhao/yi/D2H/logs/rsl_rl/reorient/2026-03-31_19-16-43/model_2250.pt --grasp_path /home/yizhao/yi/D2H/grasp_data.npy;
 ```