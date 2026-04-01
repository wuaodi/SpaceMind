#!/usr/bin/env python3
# 文字世界模型 - 执行动作前用大模型做假设推演，提出候选动作并预测执行后状态

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_WORLD_MODEL_FALLBACK_REASON = "world-model-fallback"


class TextWorldModel:
    """文字世界模型：在执行动作前，用大模型做假设推演，提出候选动作并预测执行后状态。"""

    def __init__(self, model_client):
        self.model_client = model_client

    async def imagine_and_select(
        self,
        image_base64: str,
        current_state_text: str,
        history_context: str,
        task_description: str,
        available_tools: list[str],
        planner_system_prompt: str,
        selector_system_prompt: str,
        planner_trace_metadata: Optional[dict[str, Any]] = None,
        selector_trace_metadata: Optional[dict[str, Any]] = None,
    ) -> dict:
        """
        提出 3 个候选动作，预测执行后状态，选出最优动作。
        返回: {"tool_name": str, "tool_args": dict, "reasoning": str, "candidates": list}
        """
        normalized_tools = self._normalize_available_tools(available_tools)
        user_text = f"""Task: {task_description}

Current state:
{current_state_text}

Available tool actions:
{json.dumps(normalized_tools, ensure_ascii=False)}

Propose 3 candidate actions, predict outcomes, and select the best. Output JSON only."""

        msg = await self.model_client.chat_with_image(
            image_base64=image_base64,
            system_prompt=planner_system_prompt,
            user_text=user_text,
            history_context=history_context,
            trace_metadata=planner_trace_metadata,
        )
        if not msg or not msg.content:
            logger.warning("imagine_and_select: no response from model")
            return await self._fallback_select(
                image_base64,
                current_state_text,
                history_context,
                task_description,
                selector_system_prompt,
                selector_trace_metadata,
            )

        raw = msg.content.strip()
        # 尝试提取 JSON（可能被 markdown 包裹）
        json_str = self._extract_json(raw)
        if json_str:
            try:
                data = json.loads(json_str)
                return self._parse_and_return(data, normalized_tools)
            except json.JSONDecodeError as e:
                logger.warning("imagine_and_select: JSON parse failed: %s", e)

        return await self._fallback_select(
            image_base64,
            current_state_text,
            history_context,
            task_description,
            normalized_tools,
            selector_system_prompt,
            selector_trace_metadata,
        )

    def _extract_json(self, text: str) -> Optional[str]:
        """从文本中提取 JSON 块"""
        # 尝试 ```json ... ``` 包裹
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            return m.group(1).strip()
        # 尝试 { ... } 块
        start = text.find("{")
        if start >= 0:
            depth = 0
            for i, c in enumerate(text[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
        return text.strip()

    def _clean_action_name(self, action_name: Any) -> str:
        text = str(action_name or "").strip()
        if "(" in text:
            text = text.split("(", 1)[0].strip()
        return text

    def _normalize_available_tools(self, available_tools: list[str]) -> list[str]:
        normalized: list[str] = []
        for tool_name in available_tools:
            cleaned = self._clean_action_name(tool_name)
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
        return normalized

    def _safe_fallback_result(
        self,
        reason: str,
        available_tools: list[str],
        reasoning: str = "",
        candidates: Optional[list[Any]] = None,
    ) -> dict:
        candidates = candidates or []
        if "terminate_navigation" in available_tools:
            return {
                "tool_name": "terminate_navigation",
                "tool_args": {"reason": reason},
                "reasoning": reasoning or reason,
                "candidates": candidates,
            }
        if available_tools:
            return {
                "tool_name": available_tools[0],
                "tool_args": {},
                "reasoning": reasoning or reason,
                "candidates": candidates,
            }
        return {
            "tool_name": "",
            "tool_args": {},
            "reasoning": reasoning or reason,
            "candidates": candidates,
        }

    def _sanitize_selection(
        self,
        action_name: Any,
        action_args: Any,
        reasoning: str,
        candidates: list[Any],
        available_tools: list[str],
    ) -> dict:
        cleaned_action = self._clean_action_name(action_name)
        tool_args = action_args if isinstance(action_args, dict) else {}
        if not cleaned_action or cleaned_action not in available_tools:
            return self._safe_fallback_result(
                reason="world-model-invalid-action",
                available_tools=available_tools,
                reasoning=reasoning or f"Invalid world-model action: {cleaned_action or 'empty'}",
                candidates=candidates,
            )
        return {
            "tool_name": cleaned_action,
            "tool_args": tool_args,
            "reasoning": reasoning,
            "candidates": candidates,
        }

    def _parse_and_return(self, data: dict, available_tools: list[str]) -> dict:
        if "candidates" not in data or not data["candidates"]:
            raise ValueError("No candidates in response")
        try:
            selected_idx = int(data.get("selected", 0))
        except (TypeError, ValueError):
            selected_idx = 0
        if selected_idx >= len(data["candidates"]):
            selected_idx = 0
        c = data["candidates"][selected_idx]
        return self._sanitize_selection(
            action_name=c.get("action", ""),
            action_args=c.get("args", {}),
            reasoning=str(data.get("reasoning", "")),
            candidates=data.get("candidates", []),
            available_tools=available_tools,
        )

    async def _fallback_select(
        self,
        image_base64: str,
        current_state_text: str,
        history_context: str,
        task_description: str,
        available_tools: list[str],
        system_prompt: str,
        trace_metadata: Optional[dict[str, Any]] = None,
    ) -> dict:
        """JSON 解析失败时，直接让模型选一个动作"""
        logger.info("imagine_and_select: using fallback direct selection")
        user_text = f"""Task: {task_description}

Current state:
{current_state_text}

Available tool actions:
{json.dumps(available_tools, ensure_ascii=False)}

Select one action. Output JSON only."""

        msg = await self.model_client.chat_with_image(
            image_base64=image_base64,
            system_prompt=system_prompt,
            user_text=user_text,
            history_context=history_context,
            trace_metadata=trace_metadata,
        )
        if not msg or not msg.content:
            return self._safe_fallback_result(
                reason=DEFAULT_WORLD_MODEL_FALLBACK_REASON + "-no-response",
                available_tools=available_tools,
                reasoning="fallback: no response",
            )

        raw = msg.content.strip()
        json_str = self._extract_json(raw)
        if json_str:
            try:
                data = json.loads(json_str)
                return self._sanitize_selection(
                    action_name=data.get("action", ""),
                    action_args=data.get("args", {}),
                    reasoning=str(data.get("reasoning", "")),
                    candidates=[data],
                    available_tools=available_tools,
                )
            except json.JSONDecodeError:
                pass
        return self._safe_fallback_result(
            reason=DEFAULT_WORLD_MODEL_FALLBACK_REASON + "-parse-failed",
            available_tools=available_tools,
            reasoning="fallback: parse failed",
        )


if __name__ == "__main__":
    import asyncio
    import os
    from dotenv import load_dotenv
    from models.model_client import ModelClient

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    async def demo():
        client = ModelClient(
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            max_tokens=1024,
        )
        wm = TextWorldModel(client)
        state = """Current state:
- Distance to target: 20.18m
- Target relative position: x=20.18m, y=0.00m, z=5.29m
- Target in view: yes"""
        result = await wm.imagine_and_select(
            image_base64="",  # 无图时可能失败，仅演示接口
            current_state_text=state,
            history_context="First step.",
            task_description="Approach target spacecraft safely.",
            available_tools=["set_position", "set_attitude", "terminate_navigation"],
            planner_system_prompt="Return 3 candidate actions as JSON.",
            selector_system_prompt="Return 1 action as JSON.",
        )
        print(result)

    asyncio.run(demo())
