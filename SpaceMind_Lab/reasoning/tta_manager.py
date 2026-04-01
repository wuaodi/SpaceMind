#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from models.model_client import ModelClient

from .skill_manager import SkillRuntime


DECISION_TYPES = {"create", "overlay", "rewrite", "disable", "no_change"}
SKILL_NAME_RE = re.compile(r"[^a-z0-9]+")
DEFAULT_TTA_TOPK = 3
DEFAULT_TTA_HISTORY_WINDOW = 5
DEFAULT_TTA_ALLOW_REWRITE = True


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) >= 3}


def _normalize_skill_name(raw: str, fallback: str) -> str:
    lowered = raw.strip().lower()
    normalized = SKILL_NAME_RE.sub("-", lowered).strip("-")
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized[:64] or fallback


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
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


class TTARuntime:
    def __init__(
        self,
        project_root: Path,
        model_client: ModelClient,
        task_name: str,
        run_mode: str,
        workspace_path: Path,
    ):
        self.project_root = Path(project_root)
        self.model_client = model_client
        self.task_name = task_name
        self.run_mode = run_mode
        self.workspace_path = Path(workspace_path)
        self.topk = DEFAULT_TTA_TOPK
        self.history_window = DEFAULT_TTA_HISTORY_WINDOW
        self.allow_rewrite = DEFAULT_TTA_ALLOW_REWRITE

        self.base_snapshot_dir = self.workspace_path / "base_snapshot"
        self.active_skills_dir = self.workspace_path / "active_skills"
        self.learned_skills_dir = self.workspace_path / "learned_skills"
        self.episode_summary_dir = self.workspace_path / "episode_summaries"
        self.mutation_audit_dir = self.workspace_path / "mutation_audits"
        self.meta_file = self.workspace_path / "workspace_meta.json"
        self._ensure_workspace()

    def build_skill_runtime(self) -> SkillRuntime:
        return SkillRuntime.from_project_root(
            self.project_root,
            skill_roots=[self.active_skills_dir, self.learned_skills_dir],
        )

    def get_learned_skill_count(self, skill_runtime: SkillRuntime) -> int:
        count = 0
        for skill in skill_runtime.skills.values():
            if skill.source in {"learned", "overlay"} and skill.is_active:
                count += 1
        return count

    def _ensure_workspace(self):
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.learned_skills_dir.mkdir(parents=True, exist_ok=True)
        self.episode_summary_dir.mkdir(parents=True, exist_ok=True)
        self.mutation_audit_dir.mkdir(parents=True, exist_ok=True)

        if not self.meta_file.exists():
            snapshot_id = "base_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            self._copy_tree(self.project_root / "skills", self.base_snapshot_dir)
            self._copy_tree(self.base_snapshot_dir, self.active_skills_dir)
            self._write_workspace_meta(
                {
                    "base_snapshot_id": snapshot_id,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "project_root": str(self.project_root),
                    "task_name": self.task_name,
                    "workspace_policy": "single_task_tta",
                }
            )
            return

        meta = self._load_workspace_meta()
        existing_task_name = str(meta.get("task_name", "")).strip()
        if not existing_task_name and self._workspace_has_history():
            raise RuntimeError(
                f"Legacy TTA workspace is not supported for reuse: {self.workspace_path}. "
                "This workspace has no task_name binding and already contains episode or learned skill data. "
                "Please use a new --tta_workspace for the current task."
            )
        if existing_task_name and existing_task_name != self.task_name:
            raise RuntimeError(
                f"TTA workspace {self.workspace_path} is already bound to task '{existing_task_name}', "
                f"but the current task is '{self.task_name}'. Please use a new --tta_workspace."
            )

        if not self.base_snapshot_dir.exists():
            self._copy_tree(self.project_root / "skills", self.base_snapshot_dir)
        if not self.active_skills_dir.exists():
            self._copy_tree(self.base_snapshot_dir, self.active_skills_dir)

        meta["project_root"] = str(self.project_root)
        meta["task_name"] = self.task_name
        meta["workspace_policy"] = "single_task_tta"
        self._write_workspace_meta(meta)

    def _copy_tree(self, source: Path, target: Path):
        if not source.exists():
            raise FileNotFoundError(f"Missing skill source directory: {source}")
        shutil.copytree(source, target, dirs_exist_ok=True)

    def _write_workspace_meta(self, meta: dict[str, Any]):
        self.meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_workspace_meta(self) -> dict[str, Any]:
        return json.loads(self.meta_file.read_text(encoding="utf-8"))

    def _workspace_has_history(self) -> bool:
        if any(self.episode_summary_dir.glob("*.json")):
            return True
        if any(self.learned_skills_dir.glob("*/SKILL.md")):
            return True
        return False

    def get_base_snapshot_id(self) -> str:
        return str(self._load_workspace_meta().get("base_snapshot_id", "base_unknown"))

    def get_recent_episode_summaries(self, task_name: str, limit: int) -> list[dict[str, Any]]:
        files = sorted(self.episode_summary_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        summaries: list[dict[str, Any]] = []
        for file_path in files:
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if data.get("task") != task_name:
                continue
            summaries.append(data)
            if len(summaries) >= limit:
                break
        return list(reversed(summaries))

    def retrieve_learned_skill_names(
        self,
        skill_runtime: SkillRuntime,
        task_name: str,
        run_mode: str,
        user_task: str,
        history_text: str = "",
    ) -> list[str]:
        query_text = " ".join(
            [
                task_name,
                run_mode,
                user_task,
                history_text,
                self._recent_task_patterns_text(task_name),
            ]
        )
        query_tokens = _tokenize(query_text)

        scored: list[tuple[float, str]] = []
        for skill in skill_runtime.skills.values():
            if skill.source not in {"learned", "overlay"} or not skill.is_active:
                continue
            if not self._scope_matches_exact(skill.task_scope, task_name):
                continue
            if not self._scope_matches_exact(skill.run_mode_scope, run_mode):
                continue

            score = 4.0
            score += self._trigger_match_score(skill.trigger_condition, query_tokens)
            score += min(skill.version * 0.1, 1.0)
            scored.append((score, skill.name))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [name for _, name in scored[: self.topk]]

    def _recent_task_patterns_text(self, task_name: str) -> str:
        snippets: list[str] = []
        for summary in self.get_recent_episode_summaries(task_name, self.history_window):
            if summary.get("failure_pattern"):
                snippets.append(str(summary["failure_pattern"]))
            if summary.get("success_pattern"):
                snippets.append(str(summary["success_pattern"]))
        return "\n".join(snippets)

    def _scope_matches_exact(self, scope: list[str], value: str) -> bool:
        lowered = {str(item).strip().lower() for item in scope if str(item).strip()}
        return value.lower() in lowered

    def _trigger_match_score(self, trigger_condition: str, query_tokens: set[str]) -> float:
        if not trigger_condition:
            return 0.5
        trigger_tokens = _tokenize(trigger_condition)
        if not trigger_tokens:
            return 0.5
        overlap = len(trigger_tokens & query_tokens)
        return min(overlap * 1.5, 6.0)

    def build_episode_summary(
        self,
        *,
        task: str,
        run_mode: str,
        user_task: str,
        task_family: str = "",
        tool_profile: str = "",
        target_name: str = "",
        success: bool,
        failure_mode: str,
        termination_reason: str,
        total_steps: int,
        episode_steps: list[dict[str, Any]],
        active_skill_set: list[str],
        applied_learned_skills: list[str],
        log_file: str,
        trace_file: str,
        history_context: str,
        move_summary: str,
    ) -> dict[str, Any]:
        episode_id = Path(trace_file).stem if trace_file else datetime.now().strftime("%Y%m%d_%H%M%S")
        tool_counts = Counter(
            str(step.get("tool_name", "")).strip()
            for step in episode_steps
            if step.get("tool_called") and step.get("tool_name")
        )
        summary = {
            "episode_id": episode_id,
            "episode_index": len(self.get_recent_episode_summaries(task, 10_000)) + 1,
            "task": task,
            "run_mode": run_mode,
            "user_task": user_task,
            "task_family": task_family or task,
            "tool_profile": tool_profile,
            "target_name": target_name,
            "success": success,
            "failure_mode": failure_mode,
            "termination_reason": termination_reason,
            "total_steps": total_steps,
            "tool_counts": dict(tool_counts),
            "trajectory_summary": self._build_trajectory_summary(episode_steps, move_summary, termination_reason),
            "success_pattern": self._build_success_pattern(success, tool_counts, episode_steps, termination_reason),
            "failure_pattern": self._build_failure_pattern(success, failure_mode, episode_steps, termination_reason),
            "active_skill_set": list(active_skill_set),
            "applied_learned_skills": list(applied_learned_skills),
            "learned_skill_count": self._count_active_learned_files(),
            "applied_skill_count": len(applied_learned_skills),
            "log_file": log_file,
            "trace_file": trace_file,
            "history_context": history_context,
            "move_summary": move_summary,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        summary_path = self.episode_summary_dir / f"{episode_id}.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self._mark_recent_audits_applied(applied_learned_skills, success)
        return summary

    def _build_trajectory_summary(
        self,
        episode_steps: list[dict[str, Any]],
        move_summary: str,
        termination_reason: str,
    ) -> str:
        tools = [str(step.get("tool_name", "")).strip() for step in episode_steps if step.get("tool_called")]
        tools = [tool for tool in tools if tool]
        head = " -> ".join(tools[:8]) if tools else "no tool call"
        if len(tools) > 8:
            head += f" -> ... ({len(tools)} tools total)"
        parts = [
            f"Tool trajectory: {head}",
            f"Termination: {termination_reason or 'none'}",
        ]
        if move_summary:
            parts.append(move_summary)
        return "\n".join(parts)

    def _build_success_pattern(
        self,
        success: bool,
        tool_counts: Counter,
        episode_steps: list[dict[str, Any]],
        termination_reason: str,
    ) -> str:
        if not success:
            return ""
        if not tool_counts:
            return "Episode finished successfully without a meaningful tool pattern."
        most_common = ", ".join(f"{name} x{count}" for name, count in tool_counts.most_common(4))
        last_tools = [
            str(step.get("tool_name", "")).strip()
            for step in episode_steps[-4:]
            if step.get("tool_called") and step.get("tool_name")
        ]
        return (
            f"Successful tool pattern emphasized: {most_common}. "
            f"Episode ended with {', '.join(last_tools) or 'no terminal tool sequence'}. "
            f"Termination reason: {termination_reason or 'n/a'}."
        )

    def _build_failure_pattern(
        self,
        success: bool,
        failure_mode: str,
        episode_steps: list[dict[str, Any]],
        termination_reason: str,
    ) -> str:
        if success:
            return ""
        last_tool = ""
        for step in reversed(episode_steps):
            if step.get("tool_called") and step.get("tool_name"):
                last_tool = str(step["tool_name"])
                break
        return (
            f"Failure mode: {failure_mode or 'unknown'}. "
            f"Last effective tool: {last_tool or 'none'}. "
            f"Termination reason: {termination_reason or 'n/a'}."
        )

    def _count_active_learned_files(self) -> int:
        count = 0
        for file_path in self.learned_skills_dir.glob("*/SKILL.md"):
            try:
                skill = self._read_skill_file(file_path)
            except Exception:
                continue
            if skill.get("status", "active") != "disabled":
                count += 1
        return count

    def _read_skill_file(self, file_path: Path) -> dict[str, Any]:
        text = file_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}
        end_index = None
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                end_index = idx
                break
        if end_index is None:
            return {}
        frontmatter = yaml.safe_load("\n".join(lines[1:end_index])) or {}
        body = "\n".join(lines[end_index + 1 :]).strip()
        frontmatter["body"] = body
        return frontmatter

    async def reflect_and_apply(
        self,
        summary: dict[str, Any],
        skill_runtime: SkillRuntime,
    ) -> dict[str, Any]:
        recent_summaries = self.get_recent_episode_summaries(summary["task"], self.history_window)
        relevant_skills = self._build_relevant_skill_context(skill_runtime, summary["active_skill_set"])
        msg = await self.model_client.chat_text_only(
            system_prompt=self._reflection_system_prompt(),
            user_text=self._reflection_user_prompt(summary, recent_summaries, relevant_skills),
            trace_metadata={
                "tta_reflection": True,
                "tta_task": summary["task"],
                "tta_run_mode": summary["run_mode"],
                "tta_episode_id": summary["episode_id"],
            },
        )
        raw_content = msg.content if msg else ""
        decision = self._normalize_decision(_extract_json_object(raw_content), summary)
        applied = self._apply_mutation(decision, summary, skill_runtime)
        return applied

    def _reflection_system_prompt(self) -> str:
        return (
            "You are the TTA experience reflector for SpaceMind.\n"
            "Your job is to convert one finished episode into a reusable skill mutation for the same task only.\n"
            "Return JSON only.\n"
            "Allowed decision_type values: create, overlay, rewrite, disable, no_change.\n"
            "Prefer general rules over one-off trajectory retelling, but keep them within the current task and run mode.\n"
            "Never suggest rules that weaken safety, skip confirmation, or encourage blind aggressive motion.\n"
            "For create, overlay, and rewrite, trigger_condition, rule, and constraints must be non-empty.\n"
            "For rewrite, parent_skill must be the existing skill to overwrite.\n"
            "For overlay, parent_skill must be the existing skill being refined.\n"
            "This workspace is single-task TTA only. Do not propose cross-task or cross-mode reuse.\n"
            "If no robust generalization exists, return no_change."
        )

    def _reflection_user_prompt(
        self,
        summary: dict[str, Any],
        recent_summaries: list[dict[str, Any]],
        relevant_skills: str,
    ) -> str:
        recent_payload = []
        for item in recent_summaries[-self.history_window :]:
            recent_payload.append(
                {
                    "episode_id": item.get("episode_id"),
                    "success": item.get("success"),
                    "failure_mode": item.get("failure_mode"),
                    "termination_reason": item.get("termination_reason"),
                    "success_pattern": item.get("success_pattern"),
                    "failure_pattern": item.get("failure_pattern"),
                    "applied_learned_skills": item.get("applied_learned_skills", []),
                }
            )

        return (
            "Episode summary:\n"
            + json.dumps(summary, ensure_ascii=False, indent=2)
            + "\n\nCurrent active skill context:\n"
            + relevant_skills
            + "\n\nRecent same-task episode summaries:\n"
            + json.dumps(recent_payload, ensure_ascii=False, indent=2)
            + "\n\nThis workspace is bound to task "
            + summary["task"]
            + " and run mode "
            + summary["run_mode"]
            + ". task_scope and run_mode_scope are fixed by the system."
            + "\n\nReturn JSON with keys:\n"
            + json.dumps(
                {
                    "decision_type": "create|overlay|rewrite|disable|no_change",
                    "target_skill": "skill-name",
                    "parent_skill": "existing-skill-name-or-empty",
                    "description": "one-sentence description",
                    "trigger_condition": "when this rule should activate",
                    "reason": "why this mutation is justified",
                    "intent": "what this skill tries to achieve",
                    "rule": "main reusable rule",
                    "constraints": "boundaries and safety constraints",
                    "evidence": "brief evidence from this episode",
                    "status": "active",
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    def _build_relevant_skill_context(self, skill_runtime: SkillRuntime, skill_names: list[str]) -> str:
        lines: list[str] = []
        for skill_name in skill_names:
            skill = skill_runtime.skills.get(skill_name)
            if not skill:
                continue
            compact_text = skill.compact_prompt_text()
            if len(compact_text) > 1200:
                compact_text = compact_text[:1200] + "... [truncated]"
            lines.append(f"## {skill.name}")
            lines.append(f"Description: {skill.description}")
            if skill.source != "base":
                lines.append(f"Source: {skill.source}")
            lines.append(compact_text)
        return "\n\n".join(lines) if lines else "No active skill context available."

    def _normalize_decision(
        self,
        raw: Optional[dict[str, Any]],
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        if not raw:
            return {
                "decision_type": "no_change",
                "reason": "reflector returned no valid JSON",
            }

        decision_type = str(raw.get("decision_type", "no_change")).strip().lower()
        if decision_type not in DECISION_TYPES:
            decision_type = "no_change"

        fallback_name = f"{summary['task']}-{summary['run_mode']}-lesson"
        target_skill = _normalize_skill_name(str(raw.get("target_skill", "")), fallback_name)
        parent_skill = str(raw.get("parent_skill", "")).strip()
        if decision_type == "rewrite" and not parent_skill:
            parent_skill = target_skill

        decision = {
            "decision_type": decision_type,
            "target_skill": target_skill,
            "parent_skill": parent_skill,
            "description": str(raw.get("description", "")).strip(),
            "task_scope": [summary["task"]],
            "run_mode_scope": [summary["run_mode"]],
            "trigger_condition": str(raw.get("trigger_condition", "")).strip(),
            "reason": str(raw.get("reason", "")).strip(),
            "intent": str(raw.get("intent", "")).strip(),
            "rule": str(raw.get("rule", "")).strip(),
            "constraints": str(raw.get("constraints", "")).strip(),
            "evidence": str(raw.get("evidence", "")).strip(),
            "status": str(raw.get("status", "active")).strip() or "active",
        }
        return decision

    def _apply_mutation(
        self,
        decision: dict[str, Any],
        summary: dict[str, Any],
        skill_runtime: SkillRuntime,
    ) -> dict[str, Any]:
        gated = self._quality_gate(decision, skill_runtime)
        if gated["decision_type"] == "no_change":
            audit = self._write_mutation_audit(summary["episode_id"], gated, before_version=None, after_version=None)
            return {"decision": gated, "audit_file": str(audit), "mutation_type": "no_change"}

        if gated["decision_type"] == "disable":
            before_version, after_version = self._disable_learned_skill(gated["target_skill"])
            audit = self._write_mutation_audit(summary["episode_id"], gated, before_version, after_version)
            return {"decision": gated, "audit_file": str(audit), "mutation_type": "disable"}

        if gated["decision_type"] == "rewrite":
            before_version, after_version = self._rewrite_active_skill(gated, summary, skill_runtime)
            audit = self._write_mutation_audit(summary["episode_id"], gated, before_version, after_version)
            return {"decision": gated, "audit_file": str(audit), "mutation_type": "rewrite"}

        before_version, after_version, target_name = self._write_learned_skill(gated, summary)
        gated["target_skill"] = target_name
        audit = self._write_mutation_audit(summary["episode_id"], gated, before_version, after_version)
        return {"decision": gated, "audit_file": str(audit), "mutation_type": gated["decision_type"]}

    def _quality_gate(self, decision: dict[str, Any], skill_runtime: SkillRuntime) -> dict[str, Any]:
        decision_type = decision["decision_type"]
        if decision_type == "no_change":
            return decision
        if decision_type == "rewrite" and not self.allow_rewrite:
            return {"decision_type": "no_change", "reason": "rewrite disabled by configuration"}
        if decision_type in {"create", "overlay", "rewrite"}:
            if not decision.get("trigger_condition"):
                return {"decision_type": "no_change", "reason": "missing trigger_condition"}
            if not decision.get("rule") or not decision.get("constraints"):
                return {"decision_type": "no_change", "reason": "missing rule or constraints"}
            if self._conflicts_with_safety(decision):
                return {"decision_type": "no_change", "reason": "proposed rule conflicts with safety guidance"}
        if decision_type in {"overlay", "rewrite"}:
            parent_skill = decision.get("parent_skill", "")
            if not parent_skill or parent_skill not in skill_runtime.skills:
                return {"decision_type": "no_change", "reason": "parent skill not found"}
        if decision_type == "rewrite" and decision.get("target_skill") not in skill_runtime.skills:
            return {"decision_type": "no_change", "reason": "rewrite target skill not found"}
        if decision_type == "disable":
            if not (self.learned_skills_dir / decision["target_skill"] / "SKILL.md").exists():
                return {"decision_type": "no_change", "reason": "disable target not found"}
        if decision_type in {"create", "overlay"} and self._is_duplicate_learned_rule(decision):
            return {"decision_type": "no_change", "reason": "duplicate learned rule"}
        return decision

    def _conflicts_with_safety(self, decision: dict[str, Any]) -> bool:
        merged = " ".join(
            [
                decision.get("trigger_condition", ""),
                decision.get("rule", ""),
                decision.get("constraints", ""),
            ]
        ).lower()
        blocked_phrases = [
            "ignore safety",
            "disable safety",
            "skip safety",
            "ignore collision",
            "ignore overshoot",
            "skip final measurement",
            "always move blindly",
        ]
        return any(phrase in merged for phrase in blocked_phrases)

    def _fingerprint_decision(self, decision: dict[str, Any]) -> str:
        payload = {
            "decision_type": decision.get("decision_type", ""),
            "parent_skill": decision.get("parent_skill", ""),
            "task_scope": decision.get("task_scope", []),
            "run_mode_scope": decision.get("run_mode_scope", []),
            "trigger_condition": decision.get("trigger_condition", "").strip().lower(),
            "rule": decision.get("rule", "").strip().lower(),
            "constraints": decision.get("constraints", "").strip().lower(),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _is_duplicate_learned_rule(self, decision: dict[str, Any]) -> bool:
        target_fingerprint = self._fingerprint_decision(decision)
        for file_path in self.learned_skills_dir.glob("*/SKILL.md"):
            meta = self._read_skill_file(file_path)
            if not meta or meta.get("status", "active") == "disabled":
                continue
            existing = {
                "decision_type": meta.get("decision_type", ""),
                "parent_skill": meta.get("parent_skill", ""),
                "task_scope": meta.get("task_scope", []),
                "run_mode_scope": meta.get("run_mode_scope", []),
                "trigger_condition": meta.get("trigger_condition", ""),
                "rule": self._extract_section(meta.get("body", ""), "Rule"),
                "constraints": self._extract_section(meta.get("body", ""), "Constraints"),
            }
            if self._fingerprint_decision(existing) == target_fingerprint:
                return True
        return False

    def _extract_section(self, body: str, title: str) -> str:
        pattern = re.compile(rf"^# {re.escape(title)}\s*$", re.MULTILINE)
        match = pattern.search(body)
        if not match:
            return ""
        start = match.end()
        rest = body[start:].lstrip("\n")
        next_heading = re.search(r"^# .+$", rest, re.MULTILINE)
        if next_heading:
            return rest[: next_heading.start()].strip()
        return rest.strip()

    def _build_skill_markdown(
        self,
        *,
        name: str,
        description: str,
        source: str,
        task_scope: list[str],
        run_mode_scope: list[str],
        trigger_condition: str,
        decision_type: str,
        parent_skill: str,
        version: int,
        origin_episode_id: str,
        status: str,
        intent: str,
        rule: str,
        constraints: str,
        evidence: str,
    ) -> str:
        frontmatter = {
            "name": name,
            "description": description,
            "source": source,
            "task_scope": task_scope,
            "run_mode_scope": run_mode_scope,
            "trigger_condition": trigger_condition,
            "decision_type": decision_type,
            "parent_skill": parent_skill or None,
            "version": version,
            "origin_episode_id": origin_episode_id,
            "base_snapshot_id": self.get_base_snapshot_id(),
            "status": status,
        }
        cleaned_frontmatter = {key: value for key, value in frontmatter.items() if value is not None}
        fm_text = yaml.safe_dump(cleaned_frontmatter, allow_unicode=True, sort_keys=False).strip()
        body_lines = [
            "# Intent",
            intent.strip(),
            "",
            "# Trigger",
            trigger_condition.strip(),
            "",
            "# Rule",
            rule.strip(),
            "",
            "# Constraints",
            constraints.strip(),
            "",
            "# Evidence",
            evidence.strip(),
            "",
        ]
        body = "\n".join(body_lines).strip() + "\n"
        return f"---\n{fm_text}\n---\n{body}"

    def _write_learned_skill(
        self,
        decision: dict[str, Any],
        summary: dict[str, Any],
    ) -> tuple[int, int, str]:
        source = "overlay" if decision["decision_type"] == "overlay" else "learned"
        fallback_name = f"{summary['task']}-{summary['run_mode']}-{decision['decision_type']}"
        target_name = _normalize_skill_name(decision["target_skill"], fallback_name)
        candidate_base = target_name
        if (self.active_skills_dir / target_name / "SKILL.md").exists() and not (self.learned_skills_dir / target_name / "SKILL.md").exists():
            candidate_base = _normalize_skill_name(f"{target_name}-{source}", fallback_name)
            target_name = candidate_base
        suffix = 2
        while (self.active_skills_dir / target_name / "SKILL.md").exists() and not (self.learned_skills_dir / target_name / "SKILL.md").exists():
            target_name = _normalize_skill_name(f"{candidate_base}-{suffix}", fallback_name)
            suffix += 1
        skill_dir = self.learned_skills_dir / target_name
        before_version = 0
        if (skill_dir / "SKILL.md").exists():
            existing_meta = self._read_skill_file(skill_dir / "SKILL.md")
            before_version = int(existing_meta.get("version", 0) or 0)

        after_version = before_version + 1
        skill_dir.mkdir(parents=True, exist_ok=True)
        markdown = self._build_skill_markdown(
            name=target_name,
            description=decision["description"] or f"TTA {decision['decision_type']} skill for {summary['task']}",
            source=source,
            task_scope=decision["task_scope"],
            run_mode_scope=decision["run_mode_scope"],
            trigger_condition=decision["trigger_condition"],
            decision_type=decision["decision_type"],
            parent_skill=decision.get("parent_skill", ""),
            version=after_version,
            origin_episode_id=summary["episode_id"],
            status=decision.get("status", "active"),
            intent=decision.get("intent", "") or "Refine agent behavior based on cross-episode experience.",
            rule=decision["rule"],
            constraints=decision["constraints"],
            evidence=decision.get("evidence", "") or summary.get("trajectory_summary", ""),
        )
        (skill_dir / "SKILL.md").write_text(markdown, encoding="utf-8")
        return before_version, after_version, target_name

    def _rewrite_active_skill(
        self,
        decision: dict[str, Any],
        summary: dict[str, Any],
        skill_runtime: SkillRuntime,
    ) -> tuple[int, int]:
        target_name = decision["target_skill"]
        existing = skill_runtime.skills[target_name]
        before_version = existing.version
        after_version = before_version + 1 if before_version > 0 else 1
        skill_dir = self.active_skills_dir / target_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        markdown = self._build_skill_markdown(
            name=target_name,
            description=decision["description"] or existing.description,
            source="rewrite",
            task_scope=decision["task_scope"],
            run_mode_scope=decision["run_mode_scope"],
            trigger_condition=decision["trigger_condition"],
            decision_type="rewrite",
            parent_skill=decision.get("parent_skill", "") or target_name,
            version=after_version,
            origin_episode_id=summary["episode_id"],
            status=decision.get("status", "active"),
            intent=decision.get("intent", "") or f"Rewrite {target_name} based on validated experience.",
            rule=decision["rule"],
            constraints=decision["constraints"],
            evidence=decision.get("evidence", "") or summary.get("trajectory_summary", ""),
        )
        (skill_dir / "SKILL.md").write_text(markdown, encoding="utf-8")
        return before_version, after_version

    def _disable_learned_skill(self, target_skill: str) -> tuple[Optional[int], Optional[int]]:
        skill_file = self.learned_skills_dir / target_skill / "SKILL.md"
        if not skill_file.exists():
            return None, None
        meta = self._read_skill_file(skill_file)
        before_version = int(meta.get("version", 0) or 0)
        body = meta.get("body", "")
        meta["status"] = "disabled"
        meta["version"] = before_version + 1 if before_version > 0 else 1
        fm = yaml.safe_dump({k: v for k, v in meta.items() if k != "body"}, allow_unicode=True, sort_keys=False).strip()
        skill_file.write_text(f"---\n{fm}\n---\n{body}\n", encoding="utf-8")
        return before_version, int(meta["version"])

    def _write_mutation_audit(
        self,
        episode_id: str,
        decision: dict[str, Any],
        before_version: Optional[int],
        after_version: Optional[int],
    ) -> Path:
        audit = {
            "episode_id": episode_id,
            "decision_type": decision.get("decision_type", "no_change"),
            "target_skill": decision.get("target_skill", ""),
            "parent_skill": decision.get("parent_skill", ""),
            "base_snapshot_id": self.get_base_snapshot_id(),
            "before_version": before_version,
            "after_version": after_version,
            "reason": decision.get("reason", ""),
            "applied_next_episode": False,
            "success_after_apply": None,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        audit_path = self.mutation_audit_dir / f"{episode_id}_{decision.get('decision_type', 'no_change')}.json"
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        return audit_path

    def _mark_recent_audits_applied(self, applied_skills: list[str], success: bool):
        if not applied_skills:
            return
        audit_files = sorted(self.mutation_audit_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        remaining = set(applied_skills)
        for audit_path in audit_files:
            if not remaining:
                break
            try:
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            target_skill = audit.get("target_skill", "")
            if target_skill not in remaining:
                continue
            audit["applied_next_episode"] = True
            if audit.get("success_after_apply") is None:
                audit["success_after_apply"] = success
            audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
            remaining.remove(target_skill)
