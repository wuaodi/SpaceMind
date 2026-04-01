---
name: task-search-approach
description: Search for the target when it is initially outside the field of view, reacquire it safely, then approach to a stable front hold.
metadata:
  skill_kind: task
  routing_summary: Use when the mission starts without the target in view and requires a bounded search before approaching.
  routing_keywords:
    - search then approach
    - target not in view
    - find target first
    - reacquire then approach
    - explore then approach
    - no target in frame
---
# Task Goal

When the target is not initially visible, first regain a reliable observation, then approach to a safe near hold in front of the target spacecraft, and finally stop with `terminate_navigation(...)` when the final target-surface distance is about 2 meters.

# Stage Policy

1. Treat the mission as two stages: **search** first, **approach** second.
2. Do not begin forward motion before the target is reacquired with usable evidence.
3. Once the target is found, switch immediately from search behavior to approach behavior.

# Search Strategy

1. Sweep yaw to cover **both sides** of the starting heading, expanding outward until ±90° from start is covered.
2. Choose step sizes between 15° and 30°. Check the image and `[Sensor] LiDAR` block after every step — when LiDAR points appear or a spacecraft is visible, **stop sweeping immediately**.
3. **Yaw accumulation rule** — yaw deltas accumulate. Track your current heading offset from the start:
   - Example: three steps of `dyaw=+30` puts you at **+90°** from start.
   - From +90°, three steps of `dyaw=-30` only returns you to **0° (start)**, NOT to -90°.
   - To reach -90° from +90°, you need **six** steps of -30° (through +60, +30, 0, -30, -60, -90).
   - Returning to start is NOT the same as exploring the opposite side. You must continue past start.
4. A practical two-phase sweep: first go right to about +90° (e.g. +30, +30, +30), then keep turning left **past** the starting heading all the way to about -90° (e.g. continue with -30, -30, -30, -30, -30, -30). This ensures full ±90° coverage.
5. Complete the full ±90° range before concluding the target is not findable.

# Transition: Search → Approach

- As soon as the target is detected (visually or via the LiDAR block), **stop the search sweep**.
- Before moving forward, check whether the target is centered. If it is off to one side (the LiDAR `center_y` value or image position tells you), correct yaw first.
- Only after centering, begin the approach phase.

# Approach Strategy

1. Use the `[Sensor] LiDAR` distance reading (updated every step) to guide closing and verify the final surface-distance band.
2. Keep the target observable during approach. If it drifts to a side, correct yaw before the next forward step.
3. Reduce move size as the target gets closer, and require a fresh check near the final hold point.

# Loss During Approach

- If the target disappears from the image and LiDAR after a motion, **first reverse that motion** to return to the last known good state.
- After the reverse, re-check. If the target reappears, re-center and resume approach.
- If reversal fails, try a small yaw correction toward the last known direction (see target-recovery skill). Do **not** restart the full search sweep — that is only for the initial acquisition.

# Completion and Abort

- The mission is complete only after reacquisition and the final target-surface distance is about 2 meters.
- If structured search or recovery repeatedly fails, terminate with a clear target-lost or observation-failed reason instead of drifting indefinitely.
