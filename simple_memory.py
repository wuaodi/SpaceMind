"""
Simple history record manager
Only records tool call history, no complex judgment
"""

from typing import List, Dict, Optional, Any
from datetime import datetime


class SimpleHistoryManager:
    """Simple history record manager"""
    
    def __init__(self):
        self.history: List[Dict] = []
        self.step_count = 0
        
    def add_step(self, 
                 image_name: str,
                 analysis_result: Optional[str] = None,
                 tool_called: bool = False,
                 tool_name: Optional[str] = None,
                 tool_arguments: Optional[Dict] = None,
                 tool_result: Optional[Any] = None) -> None:
        """Add a step record"""
        self.step_count += 1
        
        step_record = {
            "step": self.step_count,
            "time": datetime.now().strftime("%H:%M:%S"),
            "image": image_name,
            "tool_called": tool_called,
            "tool_name": tool_name,
            "tool_args": tool_arguments,
            "tool_result": tool_result,
            "analysis": analysis_result
        }
        
        self.history.append(step_record)
        
        # Keep only the last 4 steps to avoid overly long context
        if len(self.history) > 4:
            self.history = self.history[-4:]
    
    def get_recent_context(self, last_n: int = 4) -> str:
        """Get recent steps context"""
        if not self.history:
            return "This is the first step."
        
        recent = self.history[-last_n:]
        
        context_lines = [
            f"=== Recent {len(recent)} steps history ===",
            f"Total steps: {self.step_count}"
        ]
        
        for record in recent:
            step_line = f"\nStep {record['step']} ({record['time']}): "
            
            if record['tool_called']:
                # Show tool name
                step_line += f"[Tool: {record['tool_name']}]"
                
                # Special handling for set_pose_change, show parameters
                if record['tool_name'] == 'set_pose_change':
                    args = record['tool_args'] or {}
                    dx, dy, dz = args.get('dx', 0), args.get('dy', 0), args.get('dz', 0)
                    dpitch, droll, dyaw = args.get('dpitch', 0), args.get('droll', 0), args.get('dyaw', 0)
                    step_line += f"\n  Parameters: dx={dx}, dy={dy}, dz={dz}"
                    if dpitch != 0 or droll != 0 or dyaw != 0:
                        step_line += f", dpitch={dpitch}, droll={droll}, dyaw={dyaw}"
                
                # Show complete tool return result
                if record.get('tool_result'):
                    result_str = str(record['tool_result'])
                    # If result is too long, truncate appropriately but keep key information
                    if len(result_str) > 500:
                        # For segmentation results, only show the front part
                        if record['tool_name'] == 'part_segmentation' and isinstance(record['tool_result'], dict):
                            step_line += f"\n  Result: Segmentation image generated (size: {record['tool_result'].get('image_width', 'N/A')}x{record['tool_result'].get('image_height', 'N/A')})"
                        else:
                            # Other long results truncated display
                            step_line += f"\n  Result: {result_str[:500]}..."
                    else:
                        # For multi-line results, add appropriate indentation
                        if '\n' in result_str:
                            # Indent each line
                            indented_result = '\n    '.join(result_str.split('\n'))
                            step_line += f"\n  Result:\n    {indented_result}"
                        else:
                            # Single line results displayed directly
                            step_line += f"\n  Result: {result_str}"
                
                # If there are analysis results also show - no truncation
                if record.get('analysis') and record['analysis']:
                    step_line += f"\n  Analysis: {record['analysis']}"
            else:
                step_line += "No tool call"
                if record.get('analysis') and record['analysis']:
                    step_line += f"\n  Analysis: {record['analysis']}"
            
            context_lines.append(step_line)
        
        context_lines.append("\n=== History end ===")
        return "\n".join(context_lines)
    
    def get_move_summary(self) -> str:
        """Get movement history summary"""
        moves = []
        for record in self.history:
            if record['tool_called'] and record['tool_name'] == 'set_pose_change':
                args = record['tool_args'] or {}
                dx, dy, dz = args.get('dx', 0), args.get('dy', 0), args.get('dz', 0)
                dpitch, droll, dyaw = args.get('dpitch', 0), args.get('droll', 0), args.get('dyaw', 0)
                if dx != 0 or dy != 0 or dz != 0:  # Only record actual movements
                    moves.append(f"Step {record['step']}: Move(dx={dx}, dy={dy}, dz={dz})")
                    if dpitch != 0 or droll != 0 or dyaw != 0:
                        moves.append(f" Attitude(p={dpitch}, r={droll}, y={dyaw})")
        
        if not moves:
            return "No movement yet"
        
        return f"Movement history:\n" + "\n".join(moves[-5:])  # Recent 5 movements 