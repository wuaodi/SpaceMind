---
name: mode-world-model-selector
description: Select a single best tool action when the world-model planner fallback is needed. Use for world_model selector calls that must output one tool action as JSON.
---
# Selector Role

Select one best next action for the current state. Choose a single tool call that best advances the task while respecting all active navigation skills.

# Output Format

Return JSON only:

```json
{
  "action": "tool_name",
  "args": {"arg": "value"},
  "reasoning": "brief reason"
}
```
