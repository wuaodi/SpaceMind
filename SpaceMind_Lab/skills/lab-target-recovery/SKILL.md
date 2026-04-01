---
name: lab-target-recovery
description: Recover a temporarily lost target in the laboratory with bounded planar search and no blind forward motion.
metadata:
  skill_kind: helper
  routing_summary: Add when the target may leave the camera view and a bounded image-only recovery policy is needed.
  routing_keywords:
    - target lost
    - recovery
    - reacquire target
    - search policy
    - bounded yaw search
---
# Recovery Policy

## Priority 1: Reverse the last action

If the target was just lost after a specific action (e.g., a yaw or position change), **first reverse that exact action** to return to the state where the target was last visible. For example:
- Lost after `set_attitude(0, 0, +15)` → try `set_attitude(0, 0, -15)`
- Lost after `set_position(dx=0.1, dy=0, dz=0)` → try `set_position(dx=-0.1, dy=0, dz=0)`

After the reverse, re-check the image. If the target reappears, re-center it and resume the mission.

## Priority 2: Small correction toward last known direction

If reversal does not help, use the last known relative direction with small yaw corrections (10-15°):

- Last known off to the right → `set_attitude(0, 0, +15)`
- Last known off to the left → `set_attitude(0, 0, -15)`

If the target is still visible but merely off-center, prefer a small yaw correction to re-center it before any forward motion.

## Priority 3: Expanding search (last resort)

If there is no reliable last known direction and reversal did not work, sweep yaw from the loss heading to cover ±90°. First go one direction (e.g. right to +90°), then turn **past** the loss heading and continue to the opposite side (e.g. -90°). Remember that yaw deltas accumulate: from +90°, three steps of -30° only returns to 0°, not to -90° — you must keep going. Check the image after every step.

# Hard Safety Rule

- Never issue another forward motion command while the target is still not visible.
- After the allowed recovery rounds are exhausted, terminate safely instead of wandering.
