---
name: core-foundation
description: Always-on operational baseline for SpaceMind. Defines shared coordinates, action policy, safety guards, observability rules, and perception workflow for all tasks.
metadata:
  skill_kind: core
---
# Body Frame

- `x`: forward positive, backward negative.
- `y`: right positive, left negative.
- Use only the axes that are actually exposed by the current platform and tool profile.
- When `dz`, `dpitch`, or `droll` are not available or not meaningful on the current robot, do not invent vertical or tilt strategies.
- When geometry tools are exposed, interpret them in the same body frame as `set_position(dx, dy, dz)`.
- The current image-side bearing of the target is not the same thing as the final task goal.

# Attitude Frame

- `dpitch < 0`: look down (tilt nose down).
- `dpitch > 0`: look up (tilt nose up).
- `dyaw > 0`: turn right (rotate clockwise when viewed from above).
- `dyaw < 0`: turn left (rotate counter-clockwise when viewed from above).
- `droll`: generally not used for target tracking; prefer pitch and yaw.
- Attitude changes are relative deltas applied to the current orientation, not absolute angles.
- Typical per-step attitude adjustment: 10-30 degrees. Avoid jumps larger than 45 degrees in a single step.
- Use only the attitude axes that are actually supported by the current platform.

# Motion Rules

- If target `x > 0`, the target is ahead, so approaching requires `dx > 0`.
- If target `x < 0`, the target is behind, so approaching requires `dx < 0`.
- If target `y > 0`, the target is to the right, so moving toward it requires `dy > 0`.
- If target `y < 0`, the target is to the left, so moving toward it requires `dy < 0`.
- Use `z` only when the current platform really supports meaningful vertical motion.
- Negative `dx` while target `x > 0` is retreat, not approach.

# Action Policy

- Call exactly one tool per step.
- Use `set_position()` for translation and `set_attitude()` for viewpoint control.
- Prefer a sense-then-act rhythm instead of repeating sensing or action without new evidence.
- If only raw image input is exposed, infer progress from target visibility, size, centering, cropping, and scene change rather than pretending you have metric truth.
- If the target is off-center or about to leave the frame, prefer `set_attitude()` before a larger translation.
- If the currently exposed tools do not include a capability, do not invent it in text; work with the available tool set.
- When the mission goal is satisfied or the mission should stop safely, call `terminate_navigation(reason)`.
- Do not call the same sensing tool more than two consecutive times without an intervening motion or termination action. If a sensing result has not changed, act on it instead of re-reading it.

# Optional Compute Tool

- If `execute_code()` is exposed, use it only when geometry, interpolation, threshold logic, or compact diagnostic reasoning genuinely benefits from computation.
- If you call `execute_code()` in this step, do not also call a motion tool in the same step.
- Let the code produce reusable output for later reasoning, not a trivial wrapper around an obvious answer.

# Safety Rules

- When the target is already close, decide whether the mission is complete before taking another forward step.
- Do not terminate from a stale estimate. Require a fresh measurement or strong fresh visual evidence near the goal.
- If a fresh measurement shows `x < 0`, treat it as overshoot, not success.
- When uncertain, choose the safer action that preserves observability and avoids collision.
- Avoid repeating the same action type more than three times without evidence of progress.
- Do not continue forward blindly after the target is lost.

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

- Use visual helper tools only when they are actually exposed in the current tool profile.
- Prefer local observation tools before moving when the needed evidence is already visible but too small or unclear.
- If no helper perception tools are exposed, rely on the raw current image plus recent action history and remain conservative.
