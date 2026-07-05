---
name: task-fly-around
description: Fly around a visible target and station directly above it at a safe distance. Use for reposition tasks that must end overhead of the target rather than in front of it.
metadata:
  skill_kind: task
  routing_summary: Use when the mission is to maneuver around the target and finish holding position directly above it.
  routing_keywords:
    - fly around
    - fly-around
    - above the target
    - overhead
    - station above
    - orbit to top
---
# Task Goal

Maneuver from the current position around the target spacecraft to a station directly ABOVE it, about 5 meters from the target center, then stop and call `terminate_navigation(...)`. The final position must be nearly overhead (within 45 degrees of the vertical axis through the target) and 3-7 meters from the target center.

# Navigation Strategy

1. This is not a straight approach. Plan an arc: gain altitude (negative dz) while managing forward distance, so you end above the target instead of in front of it.
2. Move in segments of a few meters, re-observing between segments. After each climb segment, pitch the nose DOWN (negative dpitch) so the target stays near the image center; the target will appear progressively lower in the frame as you rise.
3. Use the `[Sensor] LiDAR` block to track range to the target during the maneuver. Keep the range above 3 meters at all times to stay safe.
4. Geometry check for "overhead": in the body frame with the nose pitched down toward the target, being overhead means the target sits below you (target z>0 while the horizontal offset x/y shrinks). Reduce horizontal offset until the line of sight is close to vertical.
5. A workable sequence from a frontal start about 11 m away: climb and advance in 2-4 m steps, pitching down 15-30 degrees per stage to re-center the target, until you are looking almost straight down at it from about 5 m.
6. Do not overshoot past the target horizontally; if the target drifts toward the image edge, correct with a small translation before continuing.

# Completion

- The mission is complete only when you are holding nearly overhead of the target at about 5 meters (3-7 m acceptable) from its center, looking down at it.
- If the target leaves the field of view, recover it first (pitch/yaw sweep, small reverse translation) before continuing the arc.
