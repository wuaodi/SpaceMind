---
name: mode-react
description: Add ReAct-style reasoning/output behavior on top of the active task skills. Use when the run is in react mode and the model should think briefly in text while still preferring real function calling.
---
# ReAct Output Mode

- Follow a short `Thought -> tool call -> Observation -> next Thought` loop inside the current step.
- Prefer actual function calling for tool execution whenever possible.
- Do not describe a tool invocation in plain text when you can call the real tool directly.
- If you emit plain text reasoning, use the format `Thought: ...`.
- After the thought, if you want to state an intended plan in text, say `Next step I will ...` instead of `Action: ...`.
- Make at most one real tool call per reasoning round. Do not batch many tool calls in one response.
- Use sensing tools first when needed, then refine the next action after the observation is available.
- If you already have enough evidence to finish safely, call `terminate_navigation(...)` instead of continuing to think.
- Keep reasoning concise and action-oriented.
