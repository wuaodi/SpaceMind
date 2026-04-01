---
name: lab-navigation-policy
description: Image-first laboratory navigation policy for ground-robot experiments. Use when the task must be solved with raw current images, conservative motion, and no oracle geometry.
metadata:
  skill_kind: helper
  routing_summary: Add for laboratory image-only navigation that must avoid collisions, use small steps, and stop conservatively.
  routing_keywords:
    - laboratory navigation
    - image only
    - raw camera
    - conservative motion
    - no lidar
    - no pose
    - safe stop
---
# Lab Assumptions

- Treat `test_agv_pro` semantics as ground truth for motion:
  - `dx`: forward or backward
  - `dy`: lateral motion
  - `dyaw`: planar yaw turn
- Do not rely on `dz`, `dpitch`, or `droll` for laboratory navigation.
- Do not assume `lidar_info()`, helper visual tools, or oracle geometry exist unless they are explicitly exposed.

# Target Size Reference

The laboratory satellite model (including deployed solar panels) is roughly 50-60 cm wide. Camera field of view is wide. Measured calibration data:
- At ~2.5m away: target occupies about 50-55% of frame width
- At ~1.8m away: target occupies about 65% of frame width
- At ~1.0m away: target should occupy about 80-90% of frame width
- At ~0.5m away: target overflows the frame (too close)

If the target occupies less than ~75% of the frame width, you are still farther than 1 meter — keep approaching but start reducing step size once over 50%.

# Conservative Motion Policy

- Re-observe after each meaningful motion.
- Do not take more than two translation steps in a row without checking a fresh image again.
- If the target is clearly visible and centered:
  - target small (< 40% of frame) -> `dx` around `+0.20m`
  - target medium (40-55% of frame) -> `dx` around `+0.10m`
  - target large (55-75% of frame) -> `dx` around `+0.05m`
  - target very large (> 75% of frame) -> stop and consider termination
- If the target looks too large, cropped, or dangerously near, retreat with `dx` around `-0.05m` to `-0.10m`.
- Use `dyaw` only when the target is clearly drifting toward the edge of the frame and needs recentering. If the target is already roughly centered, do NOT turn — go straight forward.
- Use `dy` only in small amounts when lateral adjustment is truly needed.

# Decision Order

1. Decide whether the target is visible in the current fresh image.
2. If visible and already centered (within the middle third of the frame), go straight forward — do NOT turn.
3. If visible but drifting toward a frame edge, use `set_attitude()` to recenter BEFORE the next forward step.
4. Estimate target frame width percentage and pick step size accordingly.

# Safety And Termination

- Never keep moving forward after the target disappears.
- Never terminate from a stale frame or vague memory.
- Prefer stopping over gambling when the scene is ambiguous.
- A visual stop should require two consecutive fresh frames showing the target still visible, not severely cropped, and plausibly at about 1 meter from the target surface.
