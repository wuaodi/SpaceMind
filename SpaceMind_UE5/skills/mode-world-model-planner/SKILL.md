---
name: mode-world-model-planner
description: Plan with a small world-model search before acting. Use for world_model planner calls that must compare exactly three candidate actions and pick the best one in JSON.
---
# Planner Role

Before executing any action:

1. Propose exactly 3 candidate actions, each expressed as a single tool call.
2. Predict the likely outcome after each action.
3. Compare candidates on safety, efficiency, and reversibility.
4. Select the best candidate.

# Output Format

Return JSON only:

```json
{
  "candidates": [
    {
      "action": "tool_name",
      "args": {"arg": "value"},
      "predicted_outcome": "brief description",
      "risk": "low|medium|high",
      "score": 1
    }
  ],
  "selected": 0,
  "reasoning": "Why this candidate is best"
}
```
