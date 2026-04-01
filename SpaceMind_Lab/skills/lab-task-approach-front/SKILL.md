---
name: lab-task-approach-front
description: Safely approach a visible laboratory satellite target and stop in a conservative near-front hold using only current images and motion tools.
metadata:
  skill_kind: task
  routing_summary: Use when the mission is to approach the visible satellite safely from the front and stop nearby without collision.
  routing_keywords:
    - lab approach
    - approach front
    - move closer
    - stop nearby
    - image navigation
    - front hold
---
# Task Goal

Approach the visible satellite conservatively from the front, keep it observable, and stop only when fresh visual evidence suggests a safe near-front stand-off at about 1 meter from the target surface.

# Front-Approach Strategy

1. If the target is not visible, do not drive forward blindly.
2. If the target is visible but not centered, use `set_attitude()` to recenter first.
3. When the target is centered enough, advance with small `dx` steps.
4. Reduce the forward step size as the target becomes larger in the image.
5. If the target becomes cropped, suddenly very large, or visually unsafe, retreat slightly instead of advancing.

# Completion Guard

- Only stop after two consecutive fresh frames show a stable, visible, front-facing near view.
- If the target is lost during the approach, switch to a conservative recovery policy instead of continuing forward.
