---
name: task-inspection
description: Diagnose the target spacecraft using multimodal evidence. Use for inspection-diagnosis tasks.
metadata:
  skill_kind: task
  routing_summary: Use when the mission is to identify spacecraft type, visible components, and status from observation evidence.
  routing_keywords:
    - inspect
    - diagnosis
    - diagnose
    - identify components
    - summarize status
    - spacecraft type
---
# Task Goal

Collect enough evidence to describe the target spacecraft's type, visible components, status, and likely function, then call `terminate_navigation(...)` with a concise diagnosis.

# Inspection Strategy

1. Begin with the current image and the strongest currently exposed visual evidence to understand the target roughly.
2. Use local visual tools to inspect visible structure before changing viewpoint.
3. Use `set_attitude()` when a different angle is needed to disambiguate components or status.
4. Use translation only when the current viewpoint cannot support the diagnosis and a new observation position is necessary.
5. Do not assume a specific geometry tool is present; complete the diagnosis with the currently exposed evidence.
6. If the `[Sensor] LiDAR` block shows data, use it only as a standoff or safety cue, not as evidence for semantic diagnosis.
7. Once the evidence is sufficient, stop instead of over-collecting redundant observations.

# Output Expectation

Your final `terminate_navigation(reason=...)` must cover these five aspects:

1. **Type** — what class of spacecraft is this (e.g. CubeSat, large bus, etc.)
2. **Key Components** — list visible structural elements (solar panels and count, antennas, thrusters, thermal blankets, sensors, etc.)
3. **Visual Appearance** — overall shape, color, approximate size, and surface texture
4. **Function** — likely mission or purpose based on observed evidence
5. **Status** — operational condition (nominal, damaged, missing parts, etc.)

Include all five in a single concise paragraph or structured list within the termination reason string.
