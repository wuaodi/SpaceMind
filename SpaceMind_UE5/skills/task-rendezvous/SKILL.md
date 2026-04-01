---
name: task-rendezvous
description: Reach a stable front hold point near the target spacecraft. Use for visible-target approach tasks that must end at about 2m from the target surface.
metadata:
  skill_kind: task
  routing_summary: Use when the mission is to approach the target and finish at a stable front hold point.
  routing_keywords:
    - rendezvous
    - approach
    - front hold
    - stable point in front
    - reach 2m in front
---
# Task Goal

Reach a safe near hold in front of the target spacecraft, keep the target observable during the approach, then stop and call `terminate_navigation(...)` when the final target-surface distance is about 2 meters.

# Navigation Strategy

1. Start from the best currently available geometry source.
2. If the `[Sensor] LiDAR` block shows points, prefer its distance reading for coarse guidance at longer range and for final surface-distance checks.
3. If no LiDAR data is available (vision_only profile), infer progress from target size, centering, visibility, and staged re-observation.
4. When distance is measured reliably or can be estimated consistently, prefer forward step magnitudes around 20% of the current distance unless safety requires smaller motion.
5. Do not keep repeating the same cached step size for many moves without refreshing state.
6. At long range, one or two consecutive translations without re-sensing are acceptable if the target remains well centered.
7. Inside about 3m, require a fresh check before further translation.

# Completion

- The mission is complete only when the target is still in front and the final target-surface distance is about 2 meters.
- If observability breaks down before completion, recover first and then continue.
