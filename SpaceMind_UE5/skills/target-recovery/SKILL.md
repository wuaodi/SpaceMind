---
name: target-recovery
description: Recover the target when it is lost or poorly centered. Use for navigation tasks where the agent must regain visibility without wandering indefinitely.
metadata:
  skill_kind: helper
  routing_summary: Add when the mission may lose the target or needs a structured visibility recovery policy.
  routing_keywords:
    - recover
    - target lost
    - regain visibility
    - poorly centered
    - reacquire target
---
# Recovery Strategy

## Priority 1: Reverse the last action

If the target was just lost after a specific action (e.g., a yaw or position change), **first reverse that exact action** to return to the state where the target was last visible. For example:
- Lost after `set_attitude(0, 0, +20)` → try `set_attitude(0, 0, -20)`
- Lost after `set_position(dx=1, dy=0, dz=0)` → try `set_position(dx=-1, dy=0, dz=0)`

After the reverse, re-check the image and the `[Sensor] LiDAR` block. If the target reappears, re-center it and resume the mission.

## Priority 2: Small correction toward last known direction

If reversal does not help or you do not know which action caused the loss, use the last known relative direction with small yaw or pitch corrections (10-15°):

- Last known off to the right → `set_attitude(0, 0, +15)`
- Last known off to the left → `set_attitude(0, 0, -15)`
- Last known below → `set_attitude(-15, 0, 0)`
- Last known above → `set_attitude(+15, 0, 0)`

If the target is still visible but merely off-center, prefer a small yaw correction to re-center it before any forward motion.

## Priority 3: Expanding search (last resort)

If there is no reliable last known direction and reversal did not work, sweep yaw from the loss heading to cover ±90°. First go one direction (e.g. right to +90°), then turn **past** the loss heading and continue to the opposite side (e.g. -90°). Remember that yaw deltas accumulate: from +90°, three steps of -30° only returns to 0°, not to -90° — you must keep going. Check the image and the `[Sensor] LiDAR` block after every step. After the yaw sweep, try pitch adjustments (±15°) before giving up.

# Termination Guard

After repeated failed recovery attempts, stop searching indefinitely and call `terminate_navigation(...)` with a target-lost reason.
