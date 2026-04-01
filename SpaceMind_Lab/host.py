from __future__ import annotations

# SpaceMind v2 - 主控制器
# 运行方式:
#   python host.py --task rendezvous-hold-front --tool_profile lab_nav_minimal
#   python host.py --task search-then-approach --tool_profile lab_nav_minimal
#   python host.py --task inspection-diagnosis --tool_profile lab_nav_with_code --allow_code_exec
#   python host.py --task rendezvous-hold-front --model qwen3-vl-235b --enable_tta --tta_workspace tta_workspace/rendezvous-hold-front
#   python host.py --free_task "Find the satellite, approach it safely, and stop nearby."

import asyncio
import json
import logging
import os
import re
import sys
import time
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import nest_asyncio
import redis
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config.paths import EVALUATION_DIR, PROJECT_ROOT, ROOT_DOTENV_PATH, RUNTIME_LOG_DIR, TOOLS_SERVER_MODULE, TRACE_VIEWER_PATH
from config.cli_config import AVAILABLE_MODELS, Config
from models.model_client import ModelClient
from reasoning.memory_manager import SimpleHistoryManager
from reasoning.skill_gateway import SkillGateway, SkillRouteDecision
from reasoning.skill_manager import PromptProfile, SkillRuntime
from reasoning.tta_manager import TTARuntime
from reasoning.world_model_reasoner import TextWorldModel
from tools.redis_contract import KEY_LATEST_IMAGE, KEY_LATEST_LIDAR, TOPIC_IMAGE, TOPIC_POSE_CHANGE
from tools.server_tools.common import compute_lidar_surface_distance

load_dotenv(ROOT_DOTENV_PATH)
nest_asyncio.apply()

CURRENT_LOG_FILE = ""
REACT_MAX_ROUNDS_PER_STEP = 3
EVALUATION_PROTOCOL_PATH = EVALUATION_DIR / "evaluation_protocol.json"
BENCHMARK_ANNOTATION_PATH = EVALUATION_DIR / "benchmark_annotation_template.json"


def _load_evaluation_profiles() -> dict[str, dict[str, Any]]:
    if not EVALUATION_PROTOCOL_PATH.exists():
        return {}
    try:
        payload = json.loads(EVALUATION_PROTOCOL_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    profiles = payload.get("evaluation_profiles", {})
    return profiles if isinstance(profiles, dict) else {}


def _load_benchmark_annotations() -> dict[str, Any]:
    if not BENCHMARK_ANNOTATION_PATH.exists():
        return {}
    try:
        payload = json.loads(BENCHMARK_ANNOTATION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _value_in_range(value: Optional[float], bounds: Any) -> bool:
    if value is None or not isinstance(bounds, list) or len(bounds) != 2:
        return False
    try:
        lower = float(bounds[0])
        upper = float(bounds[1])
    except (TypeError, ValueError):
        return False
    return lower <= float(value) <= upper


EVALUATION_PROFILES = _load_evaluation_profiles()
BENCHMARK_ANNOTATIONS = _load_benchmark_annotations()


def setup_logging() -> logging.Logger:
    global CURRENT_LOG_FILE
    log_dir = RUNTIME_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"spacemind_{timestamp}.txt"
    CURRENT_LOG_FILE = str(log_file)

    logger = logging.getLogger("spacemind")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


logger = setup_logging()


def _extract_tool_result(tool_result) -> Any:
    if hasattr(tool_result, "content"):
        if isinstance(tool_result.content, list) and len(tool_result.content) > 0:
            item = tool_result.content[0]
            text = item.text if hasattr(item, "text") else str(item)
            try:
                return json.loads(text) if text.strip().startswith("{") else text
            except json.JSONDecodeError:
                return text
        return str(tool_result.content)
    return str(tool_result)


def _extract_message_text(message: Any) -> str:
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


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    stripped = text.strip()
    candidates = [stripped]
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _normalize_score(raw_value: Any) -> Optional[float]:
    try:
        score = float(raw_value)
    except (TypeError, ValueError):
        return None
    if score < 0:
        score = 0.0
    if score > 100:
        score = 100.0
    return round(score, 2)


def _parse_tool_arguments(raw_args: Any) -> dict:
    if isinstance(raw_args, dict):
        return raw_args
    if raw_args is None:
        return {}

    text = str(raw_args).strip()
    if not text:
        return {}

    candidates = [text]
    sanitized = re.sub(r'([:\[,]\s*)\+([0-9])', r"\1\2", text)
    sanitized = re.sub(r",\s*([}\]])", r"\1", sanitized)
    if sanitized != text:
        candidates.append(sanitized)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"Could not parse tool arguments: {text}")


def _empty_result() -> dict:
    return {
        "analysis": "",
        "tool_called": False,
        "tool_name": "",
        "tool_arguments": {},
        "tool_result": "",
        "position_change": {},
        "attitude_change": {},
        "terminate_navigation": False,
        "terminate_reason": "",
    }


def _termination_success_from_reason(reason: str) -> bool:
    lowered = (reason or "").strip().lower()
    failure_markers = [
        "target lost",
        "collision",
        "timeout",
        "failed",
        "failure",
        "unsafe",
        "overshoot",
        "abort",
        "observation-failed",
    ]
    return not any(marker in lowered for marker in failure_markers)


def _extract_react_thought(content: str) -> str:
    if not content or "Thought:" not in content:
        return ""
    tail = content.split("Thought:", 1)[1]
    for marker in ("\nNext step I will", "\nObservation:", "\nAction:", "\nTool:"):
        if marker in tail:
            tail = tail.split(marker, 1)[0]
    return tail.strip()


def _summarize_tool_observation(tool_name: str, tool_res: Any, limit: int = 500) -> str:
    if tool_name in {"image_crop", "image_zoom"} and isinstance(tool_res, str) and tool_res and not tool_res.startswith("Error:"):
        text = f"{tool_name} produced a derived local image view."
    elif tool_name == "part_segmentation" and isinstance(tool_res, dict):
        summary_payload = {key: value for key, value in tool_res.items() if key != "segmentation_image"}
        text = json.dumps(summary_payload, ensure_ascii=False)
    elif isinstance(tool_res, (dict, list)):
        text = json.dumps(tool_res, ensure_ascii=False)
    else:
        text = str(tool_res)

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[:limit] + f"... [truncated {len(text) - limit} chars]"
    return f"{tool_name}: {text}"


class MCPOpenAIClient:
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.task_config = Config.get_active_task_config()
        self.evaluation_profile = str(self.task_config.get("evaluation_profile", "")) if not Config.is_free_task() else ""
        self.tool_profile_config = Config.get_tool_profile_config()
        self.allowed_tool_names = set(self.tool_profile_config.get("allowed_tools", []))
        self._apply_runtime_env_overrides()
        mc = Config.get_model_config()
        self.model_client = ModelClient(
            api_key=mc["api_key"],
            api_keys=mc.get("api_keys"),
            api_key_labels=mc.get("api_key_labels"),
            base_url=mc["base_url"],
            model_name=mc["name"],
            max_tokens=mc["max_tokens"],
            temperature=mc["temperature"],
            top_p=mc.get("top_p", 0.2),
            tool_choice=mc.get("tool_choice", "auto"),
            default_extra_body=mc.get("extra_body"),
        )
        logger.info("Model trace file: %s", self.model_client.trace_file)
        logger.info("API key usage status: %s", self.model_client.key_usage_text_file)
        logger.info("Trace viewer: %s", TRACE_VIEWER_PATH)
        self.run_mode = Config.get_runtime_mode()
        self.tta_runtime: Optional[TTARuntime] = None
        if Config.enable_tta:
            workspace_path = Path(Config.tta_workspace)
            if not workspace_path.is_absolute():
                workspace_path = self.project_root / workspace_path
            self.tta_runtime = TTARuntime(
                project_root=self.project_root,
                model_client=self.model_client,
                task_name=Config.task,
                run_mode=self.run_mode,
                workspace_path=workspace_path,
            )
            self.skill_runtime = self.tta_runtime.build_skill_runtime()
        else:
            self.skill_runtime = SkillRuntime.from_project_root(self.project_root)
        if Config.is_free_task():
            self.user_task = Config.free_task
        else:
            self.user_task = self.skill_runtime.get_user_task(Config.task)
        self.skill_gateway = SkillGateway(
            skill_runtime=self.skill_runtime,
            model_client=self.model_client,
            tool_profile=Config.tool_profile,
            allowed_tool_names=self.allowed_tool_names,
        )
        self.skill_route_decision: Optional[SkillRouteDecision] = None
        self.auto_selected_skills: list[str] = []
        self.applied_learned_skills = self._select_tta_skills()
        self.prompt_profiles: dict[str, PromptProfile] = {}
        self.redis_client: Optional[redis.Redis] = None
        self.history = SimpleHistoryManager(max_steps=Config.memory_steps if not Config.no_memory else 0)
        if Config.no_memory:
            self.history.max_steps = 0
        self.episode_steps: list[dict[str, Any]] = []
        self.latest_evaluator_result: Optional[dict[str, Any]] = None

        self.world_model: Optional[TextWorldModel] = None

        if Config.enable_world_model:
            self.world_model = TextWorldModel(self.model_client)

        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.image_topic = TOPIC_IMAGE
        self.last_image_timestamp = ""
        self.allow_code_exec = Config.allow_code_exec and self._tool_allowed("execute_code")

    def _apply_runtime_env_overrides(self) -> None:
        os.environ["SPACEMIND_TOOL_PROFILE"] = Config.tool_profile
        if Config.target_name:
            os.environ["SPACEMIND_TARGET_SATELLITE"] = Config.target_name
        else:
            os.environ.pop("SPACEMIND_TARGET_SATELLITE", None)

    def _select_tta_skills(self) -> list[str]:
        if not self.tta_runtime:
            return []
        return self.tta_runtime.retrieve_learned_skill_names(
            self.skill_runtime,
            task_name=Config.task,
            run_mode=self.run_mode,
            user_task=self.user_task,
        )

    def _tool_allowed(self, tool_name: str) -> bool:
        return not self.allowed_tool_names or tool_name in self.allowed_tool_names

    def _build_prompt_profiles(self) -> dict[str, PromptProfile]:
        if Config.is_free_task():
            if self.run_mode in {"standard", "react"}:
                return {
                    "main": self.skill_runtime.build_freeform_profile(
                        self.user_task,
                        self.run_mode,
                        role="main",
                        routed_skill_names=self.auto_selected_skills,
                        extra_skill_names=self.applied_learned_skills,
                    ),
                }
            if self.run_mode == "world_model":
                return {
                    "world_model_planner": self.skill_runtime.build_freeform_profile(
                        self.user_task,
                        "world_model",
                        role="planner",
                        routed_skill_names=self.auto_selected_skills,
                        extra_skill_names=self.applied_learned_skills,
                    ),
                    "world_model_selector": self.skill_runtime.build_freeform_profile(
                        self.user_task,
                        "world_model",
                        role="selector",
                        routed_skill_names=self.auto_selected_skills,
                        extra_skill_names=self.applied_learned_skills,
                    ),
                }
            raise ValueError(f"Unsupported runtime mode: {self.run_mode}")

        if self.run_mode in {"standard", "react"}:
            return {
                "main": self.skill_runtime.build_profile(
                    Config.task,
                    self.run_mode,
                    role="main",
                    routed_skill_names=self.auto_selected_skills,
                    extra_skill_names=self.applied_learned_skills,
                ),
            }
        if self.run_mode == "world_model":
            return {
                "world_model_planner": self.skill_runtime.build_profile(
                    Config.task,
                    "world_model",
                    role="planner",
                    routed_skill_names=self.auto_selected_skills,
                    extra_skill_names=self.applied_learned_skills,
                ),
                "world_model_selector": self.skill_runtime.build_profile(
                    Config.task,
                    "world_model",
                    role="selector",
                    routed_skill_names=self.auto_selected_skills,
                    extra_skill_names=self.applied_learned_skills,
                ),
            }
        raise ValueError(f"Unsupported runtime mode: {self.run_mode}")

    async def _route_dynamic_skills(self) -> None:
        self.skill_route_decision = await self.skill_gateway.route(
            task_name=Config.get_runtime_task_name(),
            task_kind=Config.task_kind,
            task_family=Config.task_family,
            user_task=self.user_task,
            run_mode=self.run_mode,
        )
        self.auto_selected_skills = list(self.skill_route_decision.selected_skills)

    def _log_prompt_profiles(self):
        logger.info("Loaded %d skills from %s", len(self.skill_runtime.skills), self.skill_runtime.skill_root)
        if Config.is_free_task():
            logger.info(
                "Task config: task=%s task_kind=%s task_family=%s target_name=%s tool_profile=%s",
                Config.get_runtime_task_name(),
                Config.task_kind,
                Config.task_family,
                Config.target_name or "default",
                Config.tool_profile,
            )
            logger.info("Free task text: %s", self.user_task)
        else:
            logger.info(
                "Benchmark config: task=%s task_family=%s target_name=%s tool_profile=%s",
                Config.task,
                Config.task_family,
                Config.target_name or "default",
                Config.tool_profile,
            )
        if self.skill_route_decision:
            logger.info(
                "Skill gateway: selected=%s fallback=%s candidates=%d reason=%s",
                self.skill_route_decision.selected_skills,
                self.skill_route_decision.fallback_used,
                self.skill_route_decision.candidate_count,
                self.skill_route_decision.reason,
            )
        logger.info("Allowed tools (%s): %s", Config.tool_profile, sorted(self.allowed_tool_names))
        if Config.allow_code_exec and not self.allow_code_exec:
            logger.info("Code execution guidance disabled because execute_code is not exposed by tool profile %s", Config.tool_profile)
        if self.tta_runtime:
            logger.info("TTA workspace: %s", self.tta_runtime.workspace_path)
            logger.info("TTA applied learned skills: %s", self.applied_learned_skills)
        for key, profile in self.prompt_profiles.items():
            logger.info("Prompt profile [%s]: %s", key, profile.skill_names)

    def _build_trace_metadata(
        self,
        prompt_profile: Optional[PromptProfile] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if prompt_profile is not None:
            metadata.update(prompt_profile.to_trace_metadata())
        metadata["allow_code_exec"] = self.allow_code_exec
        metadata["code_execution_guidance_enabled"] = self.allow_code_exec
        metadata["long_term_summary"] = self.history.long_term_summary
        metadata["applied_learned_skills"] = list(self.applied_learned_skills)
        metadata["tool_profile"] = Config.tool_profile
        metadata["target_name"] = Config.target_name
        metadata["task_kind"] = Config.task_kind
        metadata["task_name"] = Config.get_runtime_task_name()
        metadata["task_family"] = Config.task_family
        metadata["evaluation_profile"] = self.evaluation_profile
        if self.skill_route_decision:
            metadata["skill_gateway"] = {
                "always_on_skills": self.skill_runtime.get_always_on_skill_names(),
                "candidate_count": self.skill_route_decision.candidate_count,
                "selected_skills": list(self.skill_route_decision.selected_skills),
                "fallback_used": self.skill_route_decision.fallback_used,
                "router_reason": self.skill_route_decision.reason,
            }
        if Config.is_free_task():
            metadata["free_task_text"] = self.user_task
        if self.latest_evaluator_result:
            metadata["evaluator"] = dict(self.latest_evaluator_result)
        if extra:
            metadata.update(extra)
        return metadata

    def _record_episode_step(self, step_index: int, image_data: dict, result: dict):
        self.episode_steps.append(
            {
                "step_index": step_index,
                "image_name": image_data.get("name", ""),
                "image_timestamp": image_data.get("timestamp", ""),
                "analysis": result.get("analysis", ""),
                "tool_called": result.get("tool_called", False),
                "tool_name": result.get("tool_name", ""),
                "tool_arguments": result.get("tool_arguments", {}),
                "tool_result": result.get("tool_result", ""),
                "terminate_navigation": result.get("terminate_navigation", False),
                "terminate_reason": result.get("terminate_reason", ""),
            }
        )

    def _collect_active_skill_set(self) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for profile in self.prompt_profiles.values():
            for skill_name in profile.skill_names:
                if skill_name in seen:
                    continue
                seen.add(skill_name)
                names.append(skill_name)
        return names

    def _read_latest_snapshot(self, key: str) -> Optional[dict]:
        if not self.redis_client:
            return None
        try:
            raw = self.redis_client.get(key)
            if not raw:
                return None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            payload = json.loads(raw)
            return payload if isinstance(payload, dict) else None
        except Exception as e:
            logger.warning("Read %s failed: %s", key, e)
            return None

    def _code_execution_guidance(self, task_desc: str) -> str:
        if not self.allow_code_exec:
            return task_desc
        lines = [
            task_desc,
            "",
            "=== Optional Code Execution Guidance ===",
            "This run allows execute_code as an optional per-step helper.",
            "Use execute_code only when the current task genuinely benefits from extra computation, such as geometry, interpolation, threshold logic, or richer diagnostic reasoning.",
            "Still call exactly one tool this step.",
            "If you call execute_code now, use a later step to convert the result into set_position() or set_attitude().",
            "Let the code produce reusable output for later reasoning, for example computed candidates, intermediate tables, compact summaries, or action suggestions.",
            "Do not use execute_code to merely wrap an obvious one-step answer in trivial code.",
            "There is no fixed output schema requirement. Prefer outputs that are easy to reuse in later steps.",
        ]
        lines.append("=== End Optional Code Execution Guidance ===")
        return "\n".join(lines)

    def _resolve_target_name(self) -> str:
        return (Config.target_name or "").strip() or "CAPSTONE"

    def _get_inspection_truth_labels(self) -> Optional[dict[str, Any]]:
        satellites = BENCHMARK_ANNOTATIONS.get("satellites", {})
        if not isinstance(satellites, dict):
            return None
        target_payload = satellites.get(self._resolve_target_name(), {})
        if not isinstance(target_payload, dict):
            return None
        inspection_labels = target_payload.get("inspection_labels")
        return inspection_labels if isinstance(inspection_labels, dict) else None

    def _collect_final_diagnosis_text(self, termination_reason: str) -> str:
        clean_reason = termination_reason.strip()
        generic_markers = {
            "task completed",
            "mission complete",
            "inspection completed",
            "inspection task completed",
        }
        if clean_reason and clean_reason.lower() not in generic_markers and len(clean_reason) >= 20:
            return clean_reason
        for step in reversed(self.episode_steps):
            analysis = str(step.get("analysis", "") or "").strip()
            if analysis:
                return analysis
        return clean_reason

    def _get_scorer_client(self) -> ModelClient:
        """Lazy-create a text-only model client for inspection scoring."""
        if not hasattr(self, "_scorer_client"):
            scorer_alias = "qwen3.5-27b"
            scorer_cfg = AVAILABLE_MODELS.get(scorer_alias)
            if not scorer_cfg:
                return self.model_client
            self._scorer_client = ModelClient(
                api_key=scorer_cfg["api_key"],
                api_keys=scorer_cfg.get("api_keys"),
                api_key_labels=scorer_cfg.get("api_key_labels"),
                base_url=scorer_cfg["base_url"],
                model_name=scorer_cfg["name"],
                max_tokens=1024,
                temperature=0.1,
            )
        return self._scorer_client

    async def _score_inspection_report(
        self,
        profile_name: str,
        diagnosis_text: str,
    ) -> dict[str, Any]:
        truth_labels = self._get_inspection_truth_labels()
        target_name = self._resolve_target_name()
        if not truth_labels:
            return {
                "goal_type": "semantic_model_score",
                "profile": profile_name,
                "target_name": target_name,
                "inspection_score": None,
                "inspection_scoring_reason": "missing_truth_labels",
            }

        scorer = self._get_scorer_client()
        message = await scorer.chat_text_only(
            system_prompt=(
                "You are a strict spacecraft-inspection evaluator.\n"
                "Compare the agent's diagnosis text against the hidden ground-truth labels.\n"
                "Score across five dimensions (total 100):\n"
                "  - type (20 pts): correct spacecraft type identification\n"
                "  - components (30 pts): key components mentioned and accurately described\n"
                "  - appearance (20 pts): visual appearance description matches ground truth\n"
                "  - function (15 pts): mission/function correctly identified\n"
                "  - status (15 pts): operational status assessment accuracy\n\n"
                "Penalize unsupported claims and major omissions.\n"
                "Return JSON only with this exact format:\n"
                "{\"score\": <0-100>, \"breakdown\": {\"type\": <0-20>, \"components\": <0-30>, "
                "\"appearance\": <0-20>, \"function\": <0-15>, \"status\": <0-15>}, "
                "\"reason\": \"<brief explanation>\"}"
            ),
            user_text=(
                f"Target name: {target_name}\n"
                f"Ground truth labels:\n{json.dumps(truth_labels, ensure_ascii=False, indent=2)}\n\n"
                f"Diagnosis text:\n{diagnosis_text or '(empty)'}"
            ),
            trace_metadata={
                "inspection_scoring": {
                    "task_name": Config.get_runtime_task_name(),
                    "target_name": target_name,
                    "evaluation_profile": profile_name,
                }
            },
        )
        raw_response = _extract_message_text(message)
        payload = _extract_json_object(raw_response)
        score = _normalize_score(payload.get("score") if isinstance(payload, dict) else None)
        reason = ""
        breakdown = {}
        if isinstance(payload, dict):
            reason = str(payload.get("reason", "") or "").strip()
            raw_bd = payload.get("breakdown")
            if isinstance(raw_bd, dict):
                breakdown = {k: raw_bd[k] for k in ("type", "components", "appearance", "function", "status") if k in raw_bd}
        if score is None:
            reason = reason or "invalid_scorer_response"
        return {
            "goal_type": "semantic_model_score",
            "profile": profile_name,
            "target_name": target_name,
            "inspection_score": score,
            "inspection_breakdown": breakdown,
            "inspection_scoring_reason": reason,
        }

    async def _evaluate_terminal_profile(self, termination_reason: str, failure_mode: str) -> Optional[dict[str, Any]]:
        profile_name = self.evaluation_profile
        profile_cfg = EVALUATION_PROFILES.get(profile_name)
        if not isinstance(profile_cfg, dict):
            return None
        goal_type = str(profile_cfg.get("goal_type", ""))

        if goal_type == "final_distance_band_success":
            lidar_snapshot = self._read_latest_snapshot(KEY_LATEST_LIDAR)
            surface_distance_m = compute_lidar_surface_distance(lidar_snapshot)
            termination_lower = (termination_reason or "").strip().lower()
            success_cfg = profile_cfg.get("success_conditions", {})
            collision_detected = failure_mode == "collision" or "collision" in termination_lower
            timed_out = failure_mode == "timeout" or "timeout" in termination_lower

            if collision_detected:
                success = False
                failure_reason = "collision"
            elif timed_out:
                success = False
                failure_reason = "timeout"
            elif surface_distance_m is None:
                success = False
                failure_reason = "missing_lidar"
            elif not _value_in_range(surface_distance_m, success_cfg.get("surface_distance_range_m")):
                success = False
                failure_reason = "surface_distance_out_of_range"
            else:
                success = True
                failure_reason = "success"

            return {
                "goal_type": goal_type,
                "success": success,
                "profile": profile_name,
                "failure_reason": failure_reason,
                "surface_distance_m": surface_distance_m,
                "collision_detected": collision_detected,
            }

        if goal_type == "semantic_model_score":
            diagnosis_text = self._collect_final_diagnosis_text(termination_reason)
            scored = await self._score_inspection_report(profile_name, diagnosis_text)
            scored["diagnosis_text"] = diagnosis_text
            return scored

        return None

    def _log_evaluator_result(self, evaluator_result: Optional[dict[str, Any]]) -> None:
        if not evaluator_result:
            return
        if evaluator_result.get("goal_type") == "semantic_model_score":
            score = evaluator_result.get("inspection_score")
            score_text = "na" if score is None else f"{float(score):.2f}"
            breakdown = evaluator_result.get("inspection_breakdown", {})
            bd_text = ", ".join(f"{k}={v}" for k, v in breakdown.items()) if breakdown else "none"
            logger.info(
                "Inspection evaluator result: profile=%s target=%s score=%s breakdown=[%s] reason=%s",
                evaluator_result.get("profile", ""),
                evaluator_result.get("target_name", ""),
                score_text,
                bd_text,
                evaluator_result.get("inspection_scoring_reason", ""),
            )
            return

        surface_text = "na"
        surface_distance = evaluator_result.get("surface_distance_m")
        if isinstance(surface_distance, (int, float)):
            surface_text = f"{surface_distance:.3f}"
        logger.info(
            "Evaluator result: success=%s profile=%s surface_distance_m=%s collision_detected=%s failure_reason=%s",
            evaluator_result.get("success", False),
            evaluator_result.get("profile", ""),
            surface_text,
            evaluator_result.get("collision_detected", False),
            evaluator_result.get("failure_reason", "unknown"),
        )

    async def _maybe_compact_history(self) -> None:
        if Config.no_memory:
            return
        payload = self.history.build_compaction_payload()
        if not payload:
            return
        self.history.apply_compaction(payload["archived_steps"])
        logger.info("Updated long-term compact history from %d archived steps", payload["archived_step_count"])

    async def connect_to_server(self):
        if Config.server:
            server_params = StdioServerParameters(command=sys.executable, args=[Config.server])
        else:
            server_params = StdioServerParameters(command=sys.executable, args=["-m", TOOLS_SERVER_MODULE])
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self.session.initialize()
        tools_result = await self.session.list_tools()
        logger.info("Connected to MCP server, tools: %s", [t.name for t in tools_result.tools])
        await self._route_dynamic_skills()
        self.prompt_profiles = self._build_prompt_profiles()
        self._log_prompt_profiles()
        logger.info("Code execution guidance enabled (allow_code_exec): %s", self.allow_code_exec)

    def setup_redis_connection(self) -> bool:
        try:
            self.redis_client = redis.Redis(host="127.0.0.1", port=6379, db=0, socket_timeout=2.0)
            self.redis_client.ping()
            logger.info("Redis connected")
            return True
        except Exception as e:
            logger.error("Redis connection failed: %s", e)
            return False

    def _decode_image_message(self, raw: Any) -> Optional[dict]:
        if not raw:
            return None
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                import ast
                data = ast.literal_eval(raw)
            image_data = {
                "name": data.get("name", ""),
                "timestamp": data.get("timestamp", ""),
                "width": data.get("width", 0),
                "height": data.get("height", 0),
                "data": data.get("data", ""),
            }
            return image_data if image_data["data"] else None
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError, ValueError):
            return None

    def _get_cached_image(self) -> Optional[dict]:
        if not self.redis_client:
            return None
        try:
            return self._decode_image_message(self.redis_client.get(KEY_LATEST_IMAGE))
        except Exception as e:
            logger.warning("Read latest_image_data failed: %s", e)
            return None

    def _claim_fresh_image(
        self,
        image_data: Optional[dict],
        *,
        min_timestamp: str = "",
        source_label: str = "",
    ) -> Optional[dict]:
        if not image_data:
            return None
        timestamp = str(image_data.get("timestamp", "") or "")
        if not timestamp:
            return None
        if self.last_image_timestamp and timestamp == self.last_image_timestamp:
            return None
        if min_timestamp and not self._is_newer_or_equal(timestamp, min_timestamp):
            return None
        self.last_image_timestamp = timestamp
        if source_label:
            logger.info("Using %s: %s", source_label, timestamp)
        return image_data

    def _request_bootstrap_capture(self) -> str:
        if not self.redis_client:
            return ""
        try:
            request_timestamp = str(int(time.time() * 1e9))
            msg = json.dumps({
                "dx": 0.0, "dy": 0.0, "dz": 0.0,
                "dpitch": 0.0, "droll": 0.0, "dyaw": 0.0,
                "timestamp": request_timestamp,
            })
            self.redis_client.publish(TOPIC_POSE_CHANGE, msg)
            logger.info("Bootstrap capture request sent: %s", request_timestamp)
            return request_timestamp
        except Exception as e:
            logger.error("Bootstrap capture request failed: %s", e)
            return ""

    def _is_newer_or_equal(self, image_timestamp: str, reference_timestamp: str) -> bool:
        if not image_timestamp:
            return False
        if not reference_timestamp:
            return True
        try:
            return int(image_timestamp) >= int(reference_timestamp)
        except ValueError:
            return image_timestamp >= reference_timestamp

    def _wait_for_live_image(self, timeout: float) -> Optional[dict]:
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(self.image_topic)
        try:
            deadline = time.time() + timeout
            while time.time() < deadline:
                msg = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if not msg:
                    continue
                image_data = self._decode_image_message(msg.get("data"))
                if image_data:
                    return image_data
        except Exception as e:
            logger.error("wait_for_image error: %s", e)
        finally:
            pubsub.unsubscribe(self.image_topic)
            pubsub.close()
        return None

    def wait_for_image(self, timeout: float = 8.0) -> Optional[dict]:
        if not self.redis_client and not self.setup_redis_connection():
            return None

        cached_image = self._claim_fresh_image(self._get_cached_image(), source_label="fresh cached image")
        if cached_image:
            return cached_image

        if not self.last_image_timestamp:
            logger.info("Requesting initial capture from host...")
            initial_request_timestamp = self._request_bootstrap_capture()
            if initial_request_timestamp:
                image_data = self._claim_fresh_image(
                    self._wait_for_live_image(timeout),
                    min_timestamp=initial_request_timestamp,
                    source_label="initial live image",
                )
                if image_data:
                    return image_data
                cached_image = self._claim_fresh_image(
                    self._get_cached_image(),
                    min_timestamp=initial_request_timestamp,
                    source_label="fresh cached image after initial capture",
                )
                if cached_image:
                    return cached_image

        logger.info("Waiting for live image...")
        image_data = self._claim_fresh_image(self._wait_for_live_image(timeout), source_label="live image")
        if image_data:
            return image_data

        cached_image = self._claim_fresh_image(self._get_cached_image(), source_label="fresh cached image after live wait")
        if cached_image:
            return cached_image

        logger.warning("No image received within %.1fs, requesting bootstrap capture", timeout)
        request_timestamp = self._request_bootstrap_capture()
        if request_timestamp:
            image_data = self._claim_fresh_image(
                self._wait_for_live_image(timeout),
                min_timestamp=request_timestamp,
                source_label="live image after bootstrap",
            )
            if image_data:
                return image_data

        cached_image = self._claim_fresh_image(
            self._get_cached_image(),
            min_timestamp=request_timestamp,
            source_label="fresh cached image after bootstrap",
        )
        if cached_image:
            return cached_image
        return None

    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        tools_result = await self.session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema or {},
                },
            }
            for t in tools_result.tools
            if self._tool_allowed(t.name)
        ]

    async def _execute_tool(self, tool_name: str, tool_args: dict) -> Any:
        if not self._tool_allowed(tool_name):
            logger.warning("Blocked tool by profile %s: %s", Config.tool_profile, tool_name)
            return f"Tool '{tool_name}' is not allowed by tool profile '{Config.tool_profile}'."
        result = await self.session.call_tool(tool_name, arguments=tool_args)
        return _extract_tool_result(result)

    def _visual_followup_images(self, tool_name: str, tool_res: Any, original_image_b64: str) -> tuple[list[str], str] | None:
        if tool_name == "part_segmentation" and isinstance(tool_res, dict):
            seg_b64 = tool_res.get("segmentation_image")
            if seg_b64:
                return [original_image_b64, seg_b64], "Original image plus segmentation image."
        if tool_name == "image_crop" and isinstance(tool_res, str) and tool_res and not tool_res.startswith("Error:"):
            return [original_image_b64, tool_res], "Original image plus cropped local view."
        if tool_name == "image_zoom" and isinstance(tool_res, str) and tool_res and not tool_res.startswith("Error:"):
            return [original_image_b64, tool_res], "Original image plus zoomed local view."
        return None

    async def _run_standard_mode(
        self, image_data: dict, tools: List[dict], task_desc: str, history_ctx: str, prompt_profile: PromptProfile
    ) -> dict:
        msg = await self.model_client.chat_with_image(
            image_base64=image_data["data"],
            system_prompt=prompt_profile.rendered_system_prompt,
            user_text=task_desc,
            tools=tools,
            history_context=history_ctx,
            trace_metadata=self._build_trace_metadata(prompt_profile),
        )
        if not msg:
            return _empty_result()

        result = _empty_result()
        result["analysis"] = msg.content or ""

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = _parse_tool_arguments(tc.function.arguments)
                except ValueError as exc:
                    logger.warning("Invalid tool arguments for %s: %s", tc.function.name, exc)
                    continue
                logger.info("🔧 Calling tool: %s", tc.function.name)
                tool_res = await self._execute_tool(tc.function.name, args)
                result["tool_called"] = True
                result["tool_name"] = tc.function.name
                result["tool_arguments"] = args
                result["tool_result"] = str(tool_res) if not isinstance(tool_res, dict) else json.dumps(tool_res)

                if tc.function.name == "set_position":
                    result["position_change"] = args
                elif tc.function.name == "set_attitude":
                    result["attitude_change"] = args
                elif tc.function.name == "terminate_navigation":
                    result["terminate_navigation"] = True
                    result["terminate_reason"] = args.get("reason", "Task completed")

                visual_followup = self._visual_followup_images(tc.function.name, tool_res, image_data["data"])
                if visual_followup:
                    followup_images, followup_prefix = visual_followup
                    followup_prompt = f"{followup_prefix} Task: {task_desc}\nDecide next action."
                    followup_msg = await self.model_client.chat_with_images(
                        image_base64_list=followup_images,
                        system_prompt=prompt_profile.rendered_system_prompt,
                        user_text=followup_prompt,
                        tools=tools,
                        history_context=history_ctx,
                        trace_metadata=self._build_trace_metadata(prompt_profile),
                    )
                    if followup_msg and followup_msg.tool_calls:
                        for stc in followup_msg.tool_calls:
                            try:
                                sargs = _parse_tool_arguments(stc.function.arguments)
                            except ValueError as exc:
                                logger.warning("Invalid follow-up tool arguments for %s: %s", stc.function.name, exc)
                                continue
                            logger.info("🔧 Calling tool (%s): %s", tc.function.name, stc.function.name)
                            sres = await self._execute_tool(stc.function.name, sargs)
                            result["tool_called"] = True
                            result["tool_name"] = stc.function.name
                            result["tool_arguments"] = sargs
                            result["tool_result"] = str(sres)
                            if stc.function.name == "set_position":
                                result["position_change"] = sargs
                            elif stc.function.name == "set_attitude":
                                result["attitude_change"] = sargs
                            elif stc.function.name == "terminate_navigation":
                                result["terminate_navigation"] = True
                                result["terminate_reason"] = sargs.get("reason", "Task completed")
                    if followup_msg and followup_msg.content:
                        result["analysis"] = f"{result['analysis']}\n\nVisual follow-up ({tc.function.name}): {followup_msg.content}"
                    break

        return result

    async def _run_react_mode(
        self, image_data: dict, tools: List[dict], task_desc: str, history_ctx: str, prompt_profile: PromptProfile
    ) -> dict:
        result = _empty_result()
        react_context_lines: list[str] = []
        analysis_lines: list[str] = []
        current_images = [image_data["data"]]
        current_prefix = ""

        for react_round in range(1, REACT_MAX_ROUNDS_PER_STEP + 1):
            round_prompt = task_desc
            if react_context_lines:
                round_prompt = (
                    f"{task_desc}\n\n"
                    "ReAct context from previous rounds:\n"
                    + "\n".join(react_context_lines)
                    + "\n\nChoose the single best next action."
                )
            if current_prefix:
                round_prompt = f"{current_prefix}\n\n{round_prompt}"

            trace_metadata = self._build_trace_metadata(prompt_profile)
            trace_metadata["react_round"] = react_round
            trace_metadata["react_max_rounds"] = REACT_MAX_ROUNDS_PER_STEP

            if len(current_images) == 1:
                msg = await self.model_client.chat_with_image(
                    image_base64=current_images[0],
                    system_prompt=prompt_profile.rendered_system_prompt,
                    user_text=round_prompt,
                    tools=tools,
                    history_context=history_ctx,
                    trace_metadata=trace_metadata,
                )
            else:
                msg = await self.model_client.chat_with_images(
                    image_base64_list=current_images,
                    system_prompt=prompt_profile.rendered_system_prompt,
                    user_text=round_prompt,
                    tools=tools,
                    history_context=history_ctx,
                    trace_metadata=trace_metadata,
                )

            if not msg:
                break

            if msg.content:
                analysis_lines.append(f"Round {react_round}: {msg.content}")
                thought = _extract_react_thought(msg.content)
                if thought:
                    logger.info("Thought[%d]: %s", react_round, thought[:200])
                    react_context_lines.append(f"Thought {react_round}: {thought}")

            if not msg.tool_calls:
                break

            tc = msg.tool_calls[0]
            if len(msg.tool_calls) > 1:
                logger.info("ReAct round %d emitted %d tool calls; executing only the first one.", react_round, len(msg.tool_calls))

            try:
                args = _parse_tool_arguments(tc.function.arguments)
            except ValueError as exc:
                logger.warning("Invalid ReAct tool arguments for %s: %s", tc.function.name, exc)
                break
            logger.info("🔧 Calling tool (ReAct round %d): %s", react_round, tc.function.name)
            tool_res = await self._execute_tool(tc.function.name, args)
            result["tool_called"] = True
            result["tool_name"] = tc.function.name
            result["tool_arguments"] = args
            result["tool_result"] = str(tool_res) if not isinstance(tool_res, dict) else json.dumps(tool_res, ensure_ascii=False)

            if tc.function.name == "set_position":
                result["position_change"] = args
            elif tc.function.name == "set_attitude":
                result["attitude_change"] = args
            elif tc.function.name == "terminate_navigation":
                result["terminate_navigation"] = True
                result["terminate_reason"] = args.get("reason", "Task completed")

            react_context_lines.append(f"Observation {react_round}: {_summarize_tool_observation(tc.function.name, tool_res)}")

            if tc.function.name in {"set_position", "set_attitude", "terminate_navigation"}:
                break

            visual_followup = self._visual_followup_images(tc.function.name, tool_res, image_data["data"])
            if visual_followup:
                current_images, current_prefix = visual_followup
            else:
                current_images = [image_data["data"]]
                current_prefix = ""

        result["analysis"] = "\n\n".join(analysis_lines).strip()
        return result

    async def _run_world_model_mode(
        self, image_data: dict, tools: List[dict], task_desc: str, history_ctx: str
    ) -> dict:
        available_tool_names = [tool["function"]["name"] for tool in tools if isinstance(tool, dict)]
        current_state_text = "No oracle pose data is available. Use the current image and recent action-observation history conservatively."

        planner_profile = self.prompt_profiles["world_model_planner"]
        selector_profile = self.prompt_profiles["world_model_selector"]
        wm_result = await self.world_model.imagine_and_select(
            image_base64=image_data["data"],
            current_state_text=current_state_text,
            history_context=history_ctx,
            task_description=task_desc,
            available_tools=available_tool_names,
            planner_system_prompt=planner_profile.rendered_system_prompt,
            selector_system_prompt=selector_profile.rendered_system_prompt,
            planner_trace_metadata=self._build_trace_metadata(planner_profile),
            selector_trace_metadata=self._build_trace_metadata(selector_profile),
        )
        tool_name = wm_result.get("tool_name", "terminate_navigation")
        tool_args = wm_result.get("tool_args", {})
        result = _empty_result()
        result["analysis"] = wm_result.get("reasoning", "")

        logger.info("🔧 Calling tool (world model): %s", tool_name)
        tool_res = await self._execute_tool(tool_name, tool_args)
        result["tool_called"] = True
        result["tool_name"] = tool_name
        result["tool_arguments"] = tool_args
        result["tool_result"] = str(tool_res) if not isinstance(tool_res, dict) else json.dumps(tool_res, ensure_ascii=False)
        if tool_name == "set_position":
            result["position_change"] = tool_args
        elif tool_name == "set_attitude":
            result["attitude_change"] = tool_args
        elif tool_name == "terminate_navigation":
            result["terminate_navigation"] = True
            result["terminate_reason"] = tool_args.get("reason", "Task completed")
        return result

    async def analyze_image_with_tools(self, image_data: dict, task: str = "") -> dict:
        if not image_data:
            return _empty_result()

        task_desc = task or self.user_task
        task_desc = self._code_execution_guidance(task_desc)
        task_desc += (
            "\n\nIMPORTANT: Before choosing any action, you MUST first analyze the current image "
            "and state whether the target is visible or not. Output your analysis as text before calling any tool."
        )
        tools = await self.get_mcp_tools()
        history_ctx = self.history.get_recent_context() if not Config.no_memory else ""

        logger.info("🤖 Current model: %s", Config.model)
        if Config.enable_world_model:
            result = await self._run_world_model_mode(
                image_data, tools, task_desc, history_ctx
            )
        elif Config.enable_react:
            prompt_profile = self.prompt_profiles["main"]
            result = await self._run_react_mode(
                image_data, tools, task_desc, history_ctx, prompt_profile
            )
        else:
            prompt_profile = self.prompt_profiles["main"]
            result = await self._run_standard_mode(
                image_data, tools, task_desc, history_ctx, prompt_profile
            )

        tool_args = result.get("tool_arguments") if result["tool_called"] else None
        if not Config.no_memory:
            self.history.add_step(
                image_name=image_data.get("name", ""),
                analysis_result=result["analysis"],
                tool_called=result["tool_called"],
                tool_name=result.get("tool_name", ""),
                tool_arguments=tool_args,
                tool_result=result.get("tool_result", ""),
            )
            await self._maybe_compact_history()
        return result

    async def cleanup(self):
        await self.exit_stack.aclose()
        if self.redis_client:
            self.redis_client.close()


async def autonomous_navigation_demo():
    Config.show_info()

    client = MCPOpenAIClient()
    await client.connect_to_server()
    if not client.setup_redis_connection():
        logger.error("Redis not available")
        return

    task = client.user_task
    step = 0
    no_tool_count = 0
    consecutive_sensing_count = 0
    last_sensing_tool = ""
    episode_success = False
    failure_mode = "unknown"
    termination_reason = ""

    MAX_EPISODE_STEPS = 50

    try:
        while True:
            step += 1
            if step > MAX_EPISODE_STEPS:
                failure_mode = "max_steps_exceeded"
                termination_reason = f"Exceeded {MAX_EPISODE_STEPS} steps without completing the task"
                episode_success = False
                logger.warning("⚠️ Step limit reached (%d), forcing termination", MAX_EPISODE_STEPS)
                break
            logger.info("📍 Step %d", step)

            image_data = client.wait_for_image()
            if not image_data:
                logger.warning("No image received")
                continue

            step_task = task
            if consecutive_sensing_count >= 3:
                step_task = (
                    f"{task}\n\n"
                    f"WARNING: You have called '{last_sensing_tool}' {consecutive_sensing_count} times consecutively without moving. "
                    f"The measurement is already confirmed. You MUST take a motion action (set_position or set_attitude) or call terminate_navigation now. "
                    f"Do NOT call any sensing tool again until you have moved."
                )
                logger.warning("⚠️ Injecting sensing-loop warning (tool=%s, count=%d)", last_sensing_tool, consecutive_sensing_count)

            result = await client.analyze_image_with_tools(image_data, task=step_task)
            client._record_episode_step(step, image_data, result)

            if result["tool_called"]:
                no_tool_count = 0
                if result["terminate_navigation"]:
                    termination_reason = result["terminate_reason"]
                    episode_success = _termination_success_from_reason(termination_reason)
                    failure_mode = "success" if episode_success else "early_termination"
                    logger.info("🏁 Episode terminated.")
                    logger.info("✅ Mission success: %s", episode_success)
                    logger.info("📝 Termination reason: %s", result["terminate_reason"])
                    logger.info("%s", client.history.get_move_summary())
                    break
                if result.get("position_change"):
                    logger.info("🚀 Position: %s", result["position_change"])
                    consecutive_sensing_count = 0
                    last_sensing_tool = ""
                elif result.get("attitude_change"):
                    logger.info("🔄 Attitude: %s", result["attitude_change"])
                    consecutive_sensing_count = 0
                    last_sensing_tool = ""
                else:
                    if result.get("tool_name") == "execute_code":
                        execute_summary = str(result.get("tool_result", "")).replace("\n", " | ")
                        logger.info("🧮 execute_code result: %s", execute_summary[:500])
                    passive = {"execute_code"}
                    current_tool = result.get("tool_name", "")
                    if current_tool in passive:
                        if current_tool == last_sensing_tool:
                            consecutive_sensing_count += 1
                        else:
                            consecutive_sensing_count = 1
                            last_sensing_tool = current_tool
                        await client.session.call_tool("set_position", arguments={"dx": 0, "dy": 0, "dz": 0})
            else:
                no_tool_count += 1
                if no_tool_count >= 5:
                    failure_mode = "no_tool_calls"
                    termination_reason = "5 steps without tool calls"
                    logger.warning("⚠️ 5 steps without tool calls, exiting")
                    break
                await client.session.call_tool("set_position", arguments={"dx": 0, "dy": 0, "dz": 0})
    except KeyboardInterrupt:
        failure_mode = "interrupted"
        termination_reason = "Interrupted by user"
        logger.warning("Interrupted")
    finally:
        await client.cleanup()
        move_summary = client.history.get_move_summary()
        logger.info("Move summary: %s", move_summary)

        if not episode_success and not termination_reason:
            termination_reason = "Episode ended without explicit termination"
            if failure_mode == "unknown":
                failure_mode = "early_stop"

        if client.tta_runtime:
            summary = client.tta_runtime.build_episode_summary(
                task=Config.task,
                run_mode=client.run_mode,
                user_task=client.user_task,
                task_family=Config.task_family,
                tool_profile=Config.tool_profile,
                target_name=Config.target_name,
                success=episode_success,
                failure_mode=failure_mode,
                termination_reason=termination_reason,
                total_steps=step,
                episode_steps=client.episode_steps,
                active_skill_set=client._collect_active_skill_set(),
                applied_learned_skills=client.applied_learned_skills,
                log_file=CURRENT_LOG_FILE,
                trace_file=str(client.model_client.trace_file),
                history_context=client.history.get_recent_context() if not Config.no_memory else "",
                move_summary=move_summary,
            )
            logger.info("TTA episode index: %d", summary["episode_index"])
            logger.info("TTA applied skill count: %d", summary["applied_skill_count"])
            logger.info("TTA learned skill count: %d", summary["learned_skill_count"])
            logger.info("TTA applied skills: %s", summary["applied_learned_skills"])
            logger.info("TTA summary file: %s", client.tta_runtime.episode_summary_dir / f"{summary['episode_id']}.json")

            mutation = await client.tta_runtime.reflect_and_apply(summary, client.skill_runtime)
            refreshed_runtime = client.tta_runtime.build_skill_runtime()
            logger.info("TTA mutation decision: %s", mutation["decision"].get("decision_type", "no_change"))
            logger.info("TTA mutation target: %s", mutation["decision"].get("target_skill", ""))
            logger.info("TTA mutation type: %s", mutation.get("mutation_type", "no_change"))
            logger.info("TTA mutation audit: %s", mutation.get("audit_file", ""))
            logger.info("TTA learned skill count (after): %d", client.tta_runtime.get_learned_skill_count(refreshed_runtime))


if __name__ == "__main__":
    Config.parse_args()
    asyncio.run(autonomous_navigation_demo())
