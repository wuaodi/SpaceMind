---
name: lab-task-search-approach
description: Find the satellite with bounded image-only search and then approach it conservatively to a safe near-front stop.
metadata:
  skill_kind: task
  routing_summary: Use when the target may be out of view and the mission is to find it first, then approach safely.
  routing_keywords:
    - search target
    - find satellite
    - reacquire then approach
    - target not visible
    - search and approach
---
# Task Goal

If the satellite is not in view, reacquire it with a bounded search. Once visible, switch immediately to approach mode and stop at about 1 meter from the target surface.

# Step-by-Step Decision (follow this EVERY step)

**At each step, you MUST first describe what you see in the current image, then answer: Is the target satellite visible?**

- **YES, target is visible** → You are in APPROACH mode. Go to "Approach Rules" below.
- **NO, target is NOT visible** → You are in SEARCH mode. Go to "Search Rules" below.

This is the most important rule: **once you see the target, NEVER go back to search turning. Stay in approach mode.**

# Search Rules (only when target is NOT visible)

1. Sweep yaw in **15°** steps to cover ±90° from starting heading. Use 15° (not 30°) so you do not skip past the target.
2. Practical sweep: first go right to +90° (six steps of +15°), then turn left past start all the way to -90° (twelve steps of -15°).
3. Remember yaw deltas accumulate: from +90°, six steps of -15° only returns to 0°, not -90°. You must keep going.
4. Check the image after every yaw step. If the target appears, **immediately stop all turning**.
5. If full ±90° sweep fails, retreat 0.10m and try one more sweep. If still not found, terminate.

# Approach Rules (only when target IS visible)

**Critical first action after finding the target: CHECK CENTERING BEFORE ANYTHING ELSE.**

1. Determine where the target is in the frame:
   - **Target in left third** of frame → call `set_attitude(dyaw=-10)` to turn left and recenter. Do NOT move forward yet.
   - **Target in right third** of frame → call `set_attitude(dyaw=+10)` to turn right and recenter. Do NOT move forward yet.
   - **Target in center third** of frame → the target is centered. Now you may move forward.
2. Only after the target is in the center third, move forward with `set_position(dx=...)`. Use step sizes from the navigation policy (start with 0.05m after a search, NOT 0.1m).
3. NEVER call `set_attitude` with a large angle (like 30°) while the target is visible. That will lose the target.
4. Reduce step size as the target grows larger. Stop when it reaches ~75% of frame width.
5. After each forward step, re-check if the target is still visible and still centered. If it drifted to one side, recenter with a small yaw before the next forward step.

# Loss During Approach

- If the target disappears after a forward step, reverse that step (e.g. dx=-0.05).
- If it disappears after a yaw, reverse the yaw.
- After reversal, re-check. If target reappears, recenter it before resuming forward motion.
- Do NOT restart the full search sweep — only do small corrections (±10° yaw).
