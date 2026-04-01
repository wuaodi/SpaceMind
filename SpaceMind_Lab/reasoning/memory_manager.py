"""
SimpleHistoryManager: history manager for agent steps.
Usage: imported by host.py as the agent memory manager.
"""

from typing import Any, Optional


class SimpleHistoryManager:
    def __init__(self, max_steps: int = 8):
        self.max_steps = max_steps
        self._steps: list[dict[str, Any]] = []
        self._archived_lines: list[str] = []
        self._archived_move_lines: list[str] = []
        self.archived_step_count = 0
        self.long_term_summary = ""
        self.total_steps_seen = 0

    def add_step(
        self,
        image_name: str = "",
        analysis_result: str = "",
        tool_called: bool = False,
        tool_name: str = "",
        tool_arguments: Optional[dict] = None,
        tool_result: str = "",
    ):
        self.total_steps_seen += 1
        step = {
            "step_index": self.total_steps_seen,
            "image_name": image_name,
            "analysis_result": analysis_result,
            "tool_called": tool_called,
            "tool_name": tool_name,
            "tool_arguments": tool_arguments or {},
            "tool_result": tool_result,
        }
        self._steps.append(step)

    def _normalize_text(self, text: str) -> str:
        return " ".join(str(text).split())

    def _format_args(self, args: dict) -> str:
        if not args:
            return ""
        parts = [f"{k}={self._summarize_value(k, v)}" for k, v in args.items()]
        return ", ".join(parts)

    def _summarize_text(self, text: str, limit: int = 320) -> str:
        compact = self._normalize_text(text)
        if len(compact) <= limit:
            return compact
        return compact[:limit] + f"... [truncated {len(compact) - limit} chars]"

    def _summarize_value(self, key: str, value: Any) -> str:
        if isinstance(value, str):
            limit = 160 if key == "code" else 320
            return self._summarize_text(value, limit=limit)
        return str(value)

    def _summarize_tool_result(self, tool_name: str, tool_result: str) -> str:
        if not tool_result:
            return ""

        if tool_name in ("image_zoom", "image_crop") and not str(tool_result).startswith("Error:"):
            return f"{tool_name} image returned (base64 PNG, {len(str(tool_result))} chars)"

        normalized = self._normalize_text(tool_result)
        if '"segmentation_image"' in normalized:
            return "part_segmentation returned segmentation image payload"

        return self._summarize_text(normalized)

    def _format_step_lines(self, step: dict[str, Any], detailed: bool = False) -> list[str]:
        step_index = step.get("step_index", 0)
        tool_name = step.get("tool_name", "")
        tool_args = step.get("tool_arguments") or {}
        tool_result = step.get("tool_result", "")
        analysis_result = step.get("analysis_result", "")

        args_str = self._format_args(tool_args)
        if tool_name:
            if args_str:
                lines = [f"Step {step_index}: [Tool: {tool_name}] Args: {args_str}"]
            else:
                lines = [f"Step {step_index}: [Tool: {tool_name}]"]
            if detailed and analysis_result:
                lines.append(f"  Analysis: {self._normalize_text(analysis_result)}")
            if tool_result:
                lines.append(f"  Result: {self._summarize_tool_result(tool_name, tool_result)}")
            return lines

        lines = [f"Step {step_index}: [No Tool]"]
        if analysis_result:
            if detailed:
                lines.append(f"  Analysis: {self._normalize_text(analysis_result)}")
            else:
                lines.append(f"  Analysis: {self._summarize_text(analysis_result)}")
        return lines

    def _format_move_line(self, step: dict[str, Any]) -> Optional[str]:
        if not step.get("tool_called"):
            return None
        name = step.get("tool_name", "")
        if name not in ("set_position", "set_attitude"):
            return None
        args = step.get("tool_arguments") or {}
        args_str = self._format_args(args)
        return f"Step {step.get('step_index', 0)}: {name}({args_str})"

    def _render_long_term_summary(self) -> str:
        if not self._archived_lines:
            return "No long-term history yet."

        lines = [
            f"Archived steps: {self.archived_step_count}",
            "Compact archived history:",
        ]
        lines.extend(self._archived_lines)
        return "\n".join(lines)

    def needs_compaction(self) -> bool:
        return self.max_steps > 0 and len(self._steps) > self.max_steps

    def build_compaction_payload(self) -> Optional[dict[str, Any]]:
        if not self.needs_compaction():
            return None

        keep_recent = min(self.max_steps, len(self._steps))
        archived = self._steps[:-keep_recent]
        if not archived:
            return None

        return {
            "archived_step_count": len(archived),
            "archived_steps": archived,
        }

    def apply_compaction(self, archived_steps: list[dict[str, Any]]):
        if not archived_steps:
            return
        self.archived_step_count += len(archived_steps)
        for step in archived_steps:
            self._archived_lines.extend(self._format_step_lines(step, detailed=False))
            move_line = self._format_move_line(step)
            if move_line:
                self._archived_move_lines.append(move_line)
        self.long_term_summary = self._render_long_term_summary()
        if self.max_steps > 0:
            self._steps = self._steps[-self.max_steps:]

    def get_recent_context(self) -> str:
        recent = self._steps[-self.max_steps:] if self.max_steps > 0 else []
        lines = [
            "=== Long-Term History ===",
            self.long_term_summary or "No long-term history yet.",
            "=== End Long-Term History ===",
            "",
        ]

        if not recent:
            lines.extend([
                "=== Recent History (0 steps) ===",
                f"Total steps: {self.total_steps_seen}",
                "=== End History ===",
            ])
            return "\n".join(lines)

        lines.extend([
            f"=== Recent History ({len(recent)} steps) ===",
            f"Total steps: {self.total_steps_seen}",
        ])
        for step in recent:
            lines.extend(self._format_step_lines(step, detailed=True))
        lines.append("=== End History ===")
        return "\n".join(lines)

    def get_move_summary(self) -> str:
        moves = list(self._archived_move_lines)
        for step in self._steps:
            move_line = self._format_move_line(step)
            if move_line:
                moves.append(move_line)
        if not moves:
            return "No movement recorded."
        return "Movement history:\n" + "\n".join(moves)

    def reset(self):
        self._steps.clear()
        self._archived_lines.clear()
        self._archived_move_lines.clear()
        self.archived_step_count = 0
        self.long_term_summary = ""
        self.total_steps_seen = 0


if __name__ == "__main__":
    m = SimpleHistoryManager(max_steps=3)
    m.add_step(tool_called=True, tool_name="lidar_info", tool_result="Target center: x=20.18m, y=0.00m, z=5.29m")
    m.add_step(tool_called=True, tool_name="set_position", tool_arguments={"dx": 2.0, "dy": 0, "dz": 0.5}, tool_result="Position change completed")
    m.add_step(tool_called=True, tool_name="pose_estimation", tool_result="Distance: 18.2m, ...")
    print(m.get_recent_context())
    print()
    print(m.get_move_summary())
