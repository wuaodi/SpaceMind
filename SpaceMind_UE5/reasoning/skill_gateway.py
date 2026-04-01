#!/usr/bin/env python3

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from models.model_client import ModelClient
from reasoning.skill_manager import SkillRuntime


logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are the SpaceMind skill gateway.

Your job is to choose dynamic task-specific skills before the main agent prompt is built.

Rules:
- Use only the skills listed in the provided catalog.
- Choose exactly one primary_skill.
- primary_skill must be a task skill.
- helper_skills must only contain helper skills.
- Choose at most the provided helper limit.
- Do not invent skill names.
- Return JSON only, with no markdown fences and no extra prose.

Return exactly this shape:
{
  "primary_skill": "task-name",
  "helper_skills": ["helper-a", "helper-b"],
  "reason": "one short sentence"
}
"""


@dataclass(frozen=True)
class SkillRouteDecision:
    primary_skill: str
    helper_skills: list[str]
    selected_skills: list[str]
    reason: str
    fallback_used: bool
    candidate_count: int
    raw_response: str = ""


class SkillGateway:
    def __init__(
        self,
        skill_runtime: SkillRuntime,
        model_client: ModelClient,
        tool_profile: str,
        allowed_tool_names: set[str],
    ):
        self.skill_runtime = skill_runtime
        self.model_client = model_client
        self.tool_profile = tool_profile
        self.allowed_tool_names = sorted(allowed_tool_names)

    def build_catalog(self) -> list[dict[str, Any]]:
        catalog = []
        for skill in self.skill_runtime.get_gateway_candidate_skills():
            catalog.append(
                {
                    "name": skill.name,
                    "skill_kind": skill.skill_kind,
                    "routing_summary": skill.routing_summary,
                    "routing_keywords": skill.routing_keywords,
                    "description": skill.description,
                    "allowed_tools_hint": skill.allowed_tools or "",
                }
            )
        return catalog

    async def route(
        self,
        task_name: str,
        task_kind: str,
        task_family: str,
        user_task: str,
        run_mode: str,
    ) -> SkillRouteDecision:
        catalog = self.build_catalog()
        fallback_skills = self._fallback_skills(task_name, task_kind)
        max_helper_skills = self.skill_runtime.get_max_helper_skills()
        prompt = self._build_user_prompt(
            task_name=task_name,
            task_kind=task_kind,
            task_family=task_family,
            user_task=user_task,
            run_mode=run_mode,
            max_helper_skills=max_helper_skills,
            catalog=catalog,
        )

        message = await self.model_client.chat_text_only(
            system_prompt=ROUTER_SYSTEM_PROMPT,
            user_text=prompt,
            trace_metadata={
                "skill_gateway_request": {
                    "task_name": task_name,
                    "task_kind": task_kind,
                    "task_family": task_family,
                    "run_mode": run_mode,
                    "tool_profile": self.tool_profile,
                    "allowed_tools": list(self.allowed_tool_names),
                    "candidate_count": len(catalog),
                    "fallback_skills": list(fallback_skills),
                }
            },
        )
        raw_response = self._extract_message_text(message)
        decision_data = self._extract_json_object(raw_response)
        if not isinstance(decision_data, dict):
            logger.warning("Skill gateway returned invalid JSON, fallback to %s", fallback_skills)
            return self._build_fallback_decision(
                fallback_skills,
                reason="router_invalid_json",
                raw_response=raw_response,
                candidate_count=len(catalog),
            )

        validated = self._validate_decision(decision_data, max_helper_skills)
        if validated is None:
            logger.warning("Skill gateway returned invalid skill selection, fallback to %s", fallback_skills)
            return self._build_fallback_decision(
                fallback_skills,
                reason="router_invalid_selection",
                raw_response=raw_response,
                candidate_count=len(catalog),
            )

        return SkillRouteDecision(
            primary_skill=validated["primary_skill"],
            helper_skills=validated["helper_skills"],
            selected_skills=[validated["primary_skill"], *validated["helper_skills"]],
            reason=validated["reason"],
            fallback_used=False,
            candidate_count=len(catalog),
            raw_response=raw_response,
        )

    def _build_user_prompt(
        self,
        task_name: str,
        task_kind: str,
        task_family: str,
        user_task: str,
        run_mode: str,
        max_helper_skills: int,
        catalog: list[dict[str, Any]],
    ) -> str:
        payload = {
            "task_name": task_name,
            "task_kind": task_kind,
            "task_family": task_family,
            "run_mode": run_mode,
            "tool_profile": self.tool_profile,
            "allowed_tools": list(self.allowed_tool_names),
            "max_helper_skills": max_helper_skills,
            "user_task": user_task,
            "skill_catalog": catalog,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _fallback_skills(self, task_name: str, task_kind: str) -> list[str]:
        if task_kind == "freeform":
            return self.skill_runtime.get_free_task_fallback_dynamic_skills()
        return self.skill_runtime.get_task_fallback_dynamic_skills(task_name)

    def _build_fallback_decision(
        self,
        fallback_skills: list[str],
        reason: str,
        raw_response: str,
        candidate_count: int,
    ) -> SkillRouteDecision:
        primary_skill = fallback_skills[0]
        helper_skills = fallback_skills[1:]
        return SkillRouteDecision(
            primary_skill=primary_skill,
            helper_skills=helper_skills,
            selected_skills=[primary_skill, *helper_skills],
            reason=reason,
            fallback_used=True,
            candidate_count=candidate_count,
            raw_response=raw_response,
        )

    def _validate_decision(self, decision_data: dict[str, Any], max_helper_skills: int) -> Optional[dict[str, Any]]:
        candidate_skills = {skill.name: skill for skill in self.skill_runtime.get_gateway_candidate_skills()}
        primary_skill = decision_data.get("primary_skill")
        helper_skills = decision_data.get("helper_skills", [])
        reason = str(decision_data.get("reason", "")).strip()

        if not isinstance(primary_skill, str) or primary_skill not in candidate_skills:
            return None
        if candidate_skills[primary_skill].skill_kind != "task":
            return None
        if not isinstance(helper_skills, list) or not all(isinstance(item, str) for item in helper_skills):
            return None
        if len(helper_skills) > max_helper_skills:
            return None

        normalized_helpers: list[str] = []
        for helper_skill in helper_skills:
            if helper_skill == primary_skill:
                return None
            skill = candidate_skills.get(helper_skill)
            if skill is None or skill.skill_kind != "helper":
                return None
            if helper_skill in normalized_helpers:
                continue
            normalized_helpers.append(helper_skill)

        if not reason:
            reason = "router_selected_skills"
        return {
            "primary_skill": primary_skill,
            "helper_skills": normalized_helpers,
            "reason": reason,
        }

    def _extract_message_text(self, message: Any) -> str:
        if message is None:
            return ""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    text = getattr(item, "text", None)
                    if text:
                        parts.append(str(text))
            return "\n".join(part for part in parts if part).strip()
        return str(content).strip()

    def _extract_json_object(self, text: str) -> Optional[dict[str, Any]]:
        if not text:
            return None
        stripped = text.strip()
        candidates = [stripped]
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match:
            candidates.append(match.group(0))

        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        return None
