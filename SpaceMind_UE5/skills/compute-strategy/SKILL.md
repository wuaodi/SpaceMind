---
name: compute-strategy
description: Guide the agent to effectively use execute_code() for computation-assisted decision making during navigation and inspection tasks.
metadata:
  skill_kind: helper
  routing_summary: Add when execute_code is available and the agent may benefit from computation for trajectory planning, evidence scoring, or multi-step reasoning.
  routing_keywords:
    - execute_code
    - code execution
    - compute
    - trajectory planning
    - scoring
    - calculation
---
# When to Use `execute_code()`

Consider using `execute_code()` when:
- You need to compute a multi-step approach trajectory from the current distance.
- You want to evaluate candidate motion plans and pick the safest one.
- You want to organize or compare multiple observations into a compact decision table.
- The current diagnostic evidence needs structured scoring or ranking before termination.

Do not default to ignoring `execute_code()` when it is available; treat it as a genuine option each step.

# Rules

- Call exactly one tool per step. If you call `execute_code()` in this step, do not also call a motion tool.
- Let the code produce reusable output for later reasoning, not a trivial wrapper around an obvious answer.
- Use the computed result in a subsequent step to decide on `set_position()`, `set_attitude()`, or `terminate_navigation()`.

# Example Uses

```
# Example 1: Plan a 20%-rule approach sequence from current distance
dist = 9.8  # from lidar
steps = []
while dist > 3.0:
    dx = round(dist * 0.2, 3)
    dist -= dx
    steps.append({"dx": dx, "remaining": round(dist, 3)})
print(steps)  # use the first entry as the next move
```

```
# Example 2: Score inspection evidence before termination
evidence = {"solar_panel": "intact", "antenna": "missing", "thermal_blanket": "torn"}
score = sum(1 for v in evidence.values() if v != "intact")
print(f"Anomaly count: {score}/{len(evidence)}, recommend: {'terminate' if score > 0 else 'continue'}")
```

```
# Example 3: Compare candidate motion plans
import math
candidates = [
    {"dx": 1.0, "dy": 0, "dz": 0},
    {"dx": 0.8, "dy": 0.3, "dz": 0},
    {"dx": 0.5, "dy": 0, "dz": -0.2},
]
target = {"x": 5.0, "y": 0.5, "z": -0.1}
for c in candidates:
    remaining = math.sqrt((target["x"]-c["dx"])**2 + (target["y"]-c["dy"])**2 + (target["z"]-c["dz"])**2)
    print(f"  {c} -> remaining {remaining:.2f}m")
```
