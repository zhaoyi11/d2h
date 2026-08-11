# Clean Table OmniReset Outside-Table Termination Design

## Goal

Terminate an OmniReset episode when the object's center leaves the tabletop or
falls below its surface, while allowing the object to be lifted upward.

## Design

Add a task-local `object_outside_table` termination predicate. Transform the
object origin into the table frame with IsaacLab's `subtract_frame_transforms`,
so loaded states with translated or rotated tables are handled correctly.

The predicate returns true when either horizontal coordinate exceeds the
table's half extents (`0.4`, `0.75`) or when local z is below the tabletop
surface (`0.02`). There is no upper z bound, so lifting remains valid. Replace
the broad environment-origin safety box in `TerminationsCfg` with this term.

## Verification

Test inside, lifted, horizontal-outside, below-table, and rotated-table cases.
Run the full focused OmniReset test file and static checks.
