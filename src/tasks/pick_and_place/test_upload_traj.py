import numpy as np
import torch

traj = np.load("/home/yizhao/yi/D2H/src/tasks/pick_and_place/trajectory_keypoints.npz")
import ipdb; ipdb.set_trace()
joint_names = traj["finger_joint_names"].tolist()
joint_ids, _ = robot.find_joints(joint_names, preserve_order=True)
import ipdb; ipdb.set_trace()
i = min(round(sim_time / float(traj["dt"])), len(traj["time"]) - 1)

joint_pos = torch.tensor(traj["finger_joint_pos"][i], device=robot.device)[None]
joint_vel = torch.tensor(traj["finger_joint_vel"][i], device=robot.device)[None]
robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids)

hand_pose = torch.tensor(
    np.r_[traj["hand_root_pos"][i], traj["hand_root_quat_wxyz"][i]],
    device=robot.device,
)[None]
robot.write_root_pose_to_sim(hand_pose)

object_pose = torch.tensor(
    np.r_[traj["object_pos"][i], traj["object_quat_wxyz"][i]],
    device=robot.device,
)[None]
object.write_root_pose_to_sim(object_pose)
