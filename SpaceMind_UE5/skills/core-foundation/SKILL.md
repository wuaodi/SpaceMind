---
name: core-foundation
description: Always-on operational baseline for SpaceMind. Defines shared coordinates, action policy, safety guards, observability rules, and perception workflow for all tasks.
metadata:
  skill_kind: core
---
# Body Frame

- `x`: forward positive, backward negative.
- `y`: right positive, left negative.
- `z`: down positive, up negative.
- The `[Sensor] LiDAR` block (appended to the task description each step) uses the same body frame as `set_position(dx, dy, dz)`.

# Attitude Frame

- `dpitch < 0`: look down (tilt nose down).
- `dpitch > 0`: look up (tilt nose up).
- `dyaw > 0`: turn right (rotate clockwise when viewed from above).
- `dyaw < 0`: turn left (rotate counter-clockwise when viewed from above).
- `droll`: generally not used for target tracking; prefer pitch and yaw.
- Attitude changes are relative deltas applied to the current orientation, not absolute angles.
- Typical per-step attitude adjustment: 10-30 degrees. Avoid jumps larger than 45 degrees in a single step.

# Motion Rules

- If target `x > 0`, the target is ahead, so approaching requires `dx > 0`.
- If target `x < 0`, the target is behind, so approaching requires `dx < 0`.
- If target `y > 0`, the target is to the right, so moving toward it requires `dy > 0`.
- If target `y < 0`, the target is to the left, so moving toward it requires `dy < 0`.
- If target `z > 0`, the target is below, so moving toward it requires `dz > 0`.
- If target `z < 0`, the target is above, so moving toward it requires `dz < 0`.
- Negative `dx` while target `x > 0` is retreat, not approach.

# Action Policy

- Call exactly one tool per step.
- Use `set_position()` for translation and `set_attitude()` for viewpoint control.
- Prefer a sense-then-act rhythm instead of repeating sensing or action without new evidence.
- If the currently exposed tools do not include a capability, do not invent it in text; work with the available tool set.
- When the mission goal is satisfied or the mission should stop safely, call `terminate_navigation(reason)`.
- Do not call the same sensing tool more than two consecutive times without an intervening motion or termination action. If a sensing result has not changed, act on it instead of re-reading it.

# Optional Compute Tool

- If `execute_code()` is exposed, it is a genuine per-step option. See the `compute-strategy` skill for usage guidance and examples.
- If you call `execute_code()` in this step, do not also call a motion tool in the same step.

# Safety Rules

- When the target is already close, decide whether the mission is complete before taking another forward step.
- Do not terminate from a stale estimate. Require a fresh measurement or strong fresh visual evidence near the goal.
- If a fresh measurement shows `x < 0`, treat it as overshoot, not success.
- When uncertain, choose the safer action that preserves observability and avoids collision.
- Avoid repeating the same action type more than three times without evidence of progress.

# Progress Rules

- If recent steps are not reducing uncertainty or improving geometry, change strategy instead of repeating the same move.
- If repeated forward motion worsens observability, recover the target before continuing.
- For inspection-style tasks, avoid unnecessary translation when a viewpoint change or local visual tool can answer the question.

# Observability Rules

- Keep the target observable while moving; do not let large blind motions accumulate.
- If the target is drifting out of frame or poorly centered, prefer `set_attitude()` before making a larger translation.
- If the target is temporarily lost, use the last known direction and the currently visible context to recover it conservatively.
- After one or two failed recovery attempts, re-sense or change recovery direction instead of repeating the same motion.
- If the target remains unavailable after repeated structured recovery attempts, terminate with a target-lost or observation-failed reason instead of wandering indefinitely.

# Perception Workflow

- Start from the strongest currently exposed observation source. If no explicit range tool is exposed, rely on image quality, target size, centering, and repeated observation.
- Use `image_bright()` when visibility is questionable.
- Use `set_exposure()` when the brightness assessment shows the image is too dark or too bright.
- Use `image_zoom()` when the target is centered but needed detail is too small in the full image.
- Use `image_crop()` when you need a specific local region such as an antenna, docking interface, solar-panel root, or anomaly.
- Use `part_segmentation()` to isolate visible structure and support component reasoning.
- Use `knowledge_base()` after enough visual evidence exists to connect appearance to spacecraft type or function.
- Prefer local observation tools before moving when the needed evidence is already visible but too small or unclear.
