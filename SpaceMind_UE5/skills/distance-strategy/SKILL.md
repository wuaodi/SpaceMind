---
name: distance-strategy
description: Choose sensing cadence and translation magnitude based on estimated distance. Use for approach tasks when the model must avoid overly conservative fixed step sizes at long range.
metadata:
  skill_kind: helper
  routing_summary: Add when motion size should scale with distance instead of using a fixed conservative step.
  routing_keywords:
    - distance
    - step size
    - translation magnitude
    - staged approach
    - long range
---
# Distance-To-Step Rule

When current distance is known or can be estimated reliably from the currently exposed evidence, prefer a forward translation magnitude of about 20% of current distance unless safety requires smaller motion.

Examples:

- distance 80m -> preferred forward move about `dx = +16m`
- distance 50m -> preferred forward move about `dx = +10m`
- distance 20m -> preferred forward move about `dx = +4m`
- distance 8m -> preferred forward move about `dx = +1.6m`

Do not keep repeating the same fixed 2m move while the target is still far away.
Do not keep repeating the same cached step size for many moves without refreshing the measured distance.

# Side-Relocation Override

- This project now focuses on front-approach and search-then-approach tasks.
- Use the 20% forward-step rule mainly for front-approach phases rather than for arbitrary lateral maneuvers.
- If LiDAR or another reliable range cue says the target is still beyond the goal band, small `dx > 0` refinement moves are appropriate.
- Do not let distance guidance turn into an endless loop of repeated small moves with no fresh check.

# Open-Loop Guidance

- If the target is still far away and centered, one or two consecutive forward moves without re-sensing are acceptable.
- Do not take more than two consecutive translation steps from the same old measurement.
- If you are estimating progress from your own previous move commands rather than from a fresh sensor reading, treat that estimate as provisional.

# Sensing Bands

- Far distance: use the strongest currently exposed coarse cue to estimate progress, such as the `[Sensor] LiDAR` reading when available or stable visual size and centering trends otherwise
- Moderate or close distance: use the best currently exposed geometry or visual evidence to refine progress
- As distance shrinks, reduce move size and refresh evidence more often
- Inside about 3m, always use a fresh check before the next translation.
- If two consecutive sensing reads return the same distance, treat the measurement as confirmed and issue the next motion step immediately. Do not keep polling the same sensor without acting.
