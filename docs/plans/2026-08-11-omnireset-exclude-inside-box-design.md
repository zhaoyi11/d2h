# Clean Table OmniReset Outside-Box Reset Design

## Goal

Prevent `Clean_Table_OmniReset-v0` from resetting to dataset states where the
object is already inside the receptive box.

## Design

Keep the recorded dataset unchanged. A simulator-free task-local geometry helper
transforms every recorded object position from the environment frame into its
corresponding box frame using IsaacLab's wxyz inverse-quaternion convention. The
reset event passes the same inclusive `BOX_MIN` and `BOX_MAX` bounds used by the
task's reward and termination logic, retains only indices whose object position
is outside those bounds, and computes this index tensor once at startup.

At each reset, sample uniformly from this retained index tensor and restore the
full recorded scene state as before. This preserves the recorded relationship
between the robot, object, box, and table instead of translating individual
assets. If the dataset contains no outside-box states, fail during environment
construction with a clear `ValueError` rather than failing later in random
sampling.

No configuration option or dataset rewrite is added: the selected policy is
strictly zero inside-box reset states.

## Verification

- Add a focused test with a rotated box to prove classification is performed in
  the box frame and returns only the outside-box state.
- Verify an all-inside pool raises a clear error.
- Run the complete clean-table OmniReset test file.
- Check the real dataset retains 497 of 1,200 states.
