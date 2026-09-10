# Screw-In World-Pose Print Design

Print the measured object world pose after every screw-in physics step. Reuse the existing
`current_pose_w` sample so the diagnostic adds no simulator reads and cannot change trajectory or
recording behavior. Each flushed line includes the zero-based target sample index and the pose in
`(x, y, z, qw, qx, qy, qz)` order. Settling samples and the NPZ schema remain unchanged.
