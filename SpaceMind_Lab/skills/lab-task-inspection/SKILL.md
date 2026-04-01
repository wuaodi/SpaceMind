---
name: lab-task-inspection
description: Diagnose the laboratory target from raw image evidence, conservative viewpoint changes, and optional code support.
metadata:
  skill_kind: task
  routing_summary: Use when the mission is to inspect the visible target and produce type, component, status, and function diagnosis in the laboratory setup.
  routing_keywords:
    - inspect
    - diagnosis
    - diagnose
    - identify components
    - summarize status
    - target type
---
# Task Goal

Collect enough evidence from the current image stream to describe the target spacecraft across five dimensions, then call `terminate_navigation(...)` with a structured diagnosis.

# Inspection Strategy

1. Start from the current full image and identify the most informative visible structure first.
2. If the target is visible but not well centered, use a small `set_attitude()` change before translating.
3. Use translation only when the current view is clearly insufficient for diagnosis and a modest new angle is needed.
4. Because laboratory profiles do not expose UE5-style local visual helper tools by default, rely on raw image evidence, viewpoint change, and recent action history.
5. If `execute_code()` is exposed, use it only to organize observations or compare candidate diagnoses; do not use it as a substitute for evidence.
6. Stop once the diagnosis covers all five dimensions below without obvious unsupported claims.

# Output Expectation

Your final `terminate_navigation(reason=...)` must cover these five aspects:

1. **Type** -- what class of spacecraft is this (e.g. CubeSat, capsule, large bus, etc.)
2. **Key Components** -- list visible structural elements (solar panels and count, antennas, thrusters, thermal blankets, sensors, etc.)
3. **Visual Appearance** -- overall shape, color, approximate size, and surface texture
4. **Function** -- likely mission or purpose based on observed evidence
5. **Status** -- operational condition (nominal, damaged, missing parts, etc.)

Include all five in a single concise paragraph or structured list within the termination reason string.
