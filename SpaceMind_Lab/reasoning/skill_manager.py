#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from config.paths import FRAMEWORK_MANIFEST_PATH, PROJECT_ROOT, SKILLS_DIR


SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class SkillRuntimeError(ValueError):
    """Raised when the skill library or manifest is invalid."""


def _parse_markdown_sections(body: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_title: Optional[str] = None
    current_lines: list[str] = []

    for line in body.splitlines():
        if line.startswith("# "):
            if current_title is not None:
                sections[current_title] = "\n".join(current_lines).strip()
            current_title = line[2:].strip().lower()
            current_lines = []
            continue
        current_lines.append(line)

    if current_title is not None:
        sections[current_title] = "\n".join(current_lines).strip()
    return sections


def _normalize_scope(value: Any) -> list[str]:
    if value is None:
        return ["*"]
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return ["*"]


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: Path
    source_root: Path
    license: Optional[str] = None
    compatibility: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    allowed_tools: Optional[str] = None
    frontmatter: dict[str, Any] = field(default_factory=dict)
    body_sections: dict[str, str] = field(default_factory=dict)

    @property
    def source(self) -> str:
        value = self.frontmatter.get("source", "base")
        return value if isinstance(value, str) and value.strip() else "base"

    @property
    def status(self) -> str:
        value = self.frontmatter.get("status", "active")
        return value if isinstance(value, str) and value.strip() else "active"

    @property
    def decision_type(self) -> str:
        value = self.frontmatter.get("decision_type", "")
        return value if isinstance(value, str) else ""

    @property
    def parent_skill(self) -> str:
        value = self.frontmatter.get("parent_skill", "")
        return value if isinstance(value, str) else ""

    @property
    def trigger_condition(self) -> str:
        value = self.frontmatter.get("trigger_condition", "")
        return value if isinstance(value, str) else ""

    @property
    def version(self) -> int:
        value = self.frontmatter.get("version", 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @property
    def task_scope(self) -> list[str]:
        return _normalize_scope(self.frontmatter.get("task_scope"))

    @property
    def run_mode_scope(self) -> list[str]:
        return _normalize_scope(self.frontmatter.get("run_mode_scope"))

    @property
    def is_active(self) -> bool:
        return self.status.lower() != "disabled"

    @property
    def is_base(self) -> bool:
        return self.source == "base"

    @property
    def is_learned(self) -> bool:
        return self.source in {"learned", "overlay", "rewrite"}

    @property
    def skill_kind(self) -> str:
        value = self.metadata.get("skill_kind", "")
        return value if isinstance(value, str) else ""

    @property
    def routing_summary(self) -> str:
        value = self.metadata.get("routing_summary", "")
        return value if isinstance(value, str) else ""

    @property
    def routing_keywords(self) -> list[str]:
        raw_keywords = self.metadata.get("routing_keywords", [])
        if isinstance(raw_keywords, list):
            return [item for item in raw_keywords if isinstance(item, str)]
        return []

    def compact_prompt_text(self) -> str:
        if not self.is_learned:
            return self.body.strip()

        blocks = []
        intent = self.body_sections.get("intent", "")
        trigger = self.body_sections.get("trigger", "")
        rule = self.body_sections.get("rule", "")
        constraints = self.body_sections.get("constraints", "")

        if intent:
            blocks.append("Intent:\n" + intent)
        if trigger:
            blocks.append("Trigger:\n" + trigger)
        elif self.trigger_condition:
            blocks.append("Trigger:\n" + self.trigger_condition)
        if rule:
            blocks.append("Rule:\n" + rule)
        if constraints:
            blocks.append("Constraints:\n" + constraints)

        return "\n\n".join(blocks).strip() or self.body.strip()

    def render_block(self) -> str:
        sections = [
            f"Skill: {self.name}",
            f"Description: {self.description}",
        ]
        if self.compatibility:
            sections.append(f"Compatibility: {self.compatibility}")
        if self.is_learned:
            sections.append(f"Source: {self.source}")
            if self.parent_skill:
                sections.append(f"Parent Skill: {self.parent_skill}")
            if self.trigger_condition:
                sections.append(f"Trigger Condition: {self.trigger_condition}")
        sections.append(self.compact_prompt_text())
        return "\n".join(section for section in sections if section)


@dataclass(frozen=True)
class PromptProfile:
    task: str
    mode: str
    role: str
    user_task: str
    skill_names: list[str]
    rendered_system_prompt: str

    def to_trace_metadata(self) -> dict[str, Any]:
        return {
            "activated_skills": list(self.skill_names),
            "prompt_profile": {
                "task": self.task,
                "mode": self.mode,
                "role": self.role,
                "skill_names": list(self.skill_names),
            },
            "rendered_system_prompt": self.rendered_system_prompt,
            "user_task": self.user_task,
        }


class SkillRuntime:
    def __init__(self, skill_roots: Path | list[Path], manifest_path: Path):
        if isinstance(skill_roots, (str, Path)):
            resolved_roots = [Path(skill_roots)]
        else:
            resolved_roots = [Path(root) for root in skill_roots]
        if not resolved_roots:
            raise SkillRuntimeError("At least one skill root is required")

        self.skill_roots = resolved_roots
        self.skill_root = self.skill_roots[0]
        self.manifest_path = Path(manifest_path)
        self.skills = self._load_skills()
        self.manifest = self._load_manifest()
        self._validate_manifest()

    @classmethod
    def from_project_root(
        cls,
        project_root: Optional[Path] = None,
        skill_roots: Optional[list[Path]] = None,
    ) -> "SkillRuntime":
        root = Path(project_root) if project_root else PROJECT_ROOT
        resolved_skill_roots = skill_roots or [root / SKILLS_DIR.name]
        manifest_path = root / FRAMEWORK_MANIFEST_PATH.relative_to(PROJECT_ROOT)
        return cls(resolved_skill_roots, manifest_path)

    def get_user_task(self, task_name: str) -> str:
        try:
            return self.manifest["tasks"][task_name]["user_task"]
        except KeyError as exc:
            raise SkillRuntimeError(f"Unknown task in framework manifest: {task_name}") from exc

    def get_free_task_config(self) -> dict[str, Any]:
        config = self.manifest.get("free_task")
        if not isinstance(config, dict):
            raise SkillRuntimeError("framework_manifest.json must contain a 'free_task' object")
        return config

    def get_skill_gateway_config(self) -> dict[str, Any]:
        config = self.manifest.get("skill_gateway")
        if not isinstance(config, dict):
            raise SkillRuntimeError("framework_manifest.json must contain a 'skill_gateway' object")
        return config

    def get_always_on_skill_names(self) -> list[str]:
        return list(self.get_skill_gateway_config().get("always_on_skills", []))

    def get_gateway_candidate_skill_names(self) -> list[str]:
        return list(self.get_skill_gateway_config().get("candidate_skills", []))

    def get_gateway_candidate_skills(self) -> list[Skill]:
        return [self.skills[name] for name in self.get_gateway_candidate_skill_names()]

    def get_task_fallback_dynamic_skills(self, task_name: str) -> list[str]:
        task_cfg = self.manifest["tasks"].get(task_name)
        if not isinstance(task_cfg, dict):
            raise SkillRuntimeError(f"Unknown task in framework manifest: {task_name}")
        fallback = task_cfg.get("fallback_dynamic_skills", [])
        if not isinstance(fallback, list) or not all(isinstance(name, str) for name in fallback):
            raise SkillRuntimeError(f"Manifest task '{task_name}' has invalid fallback_dynamic_skills")
        return list(fallback)

    def get_free_task_fallback_dynamic_skills(self) -> list[str]:
        fallback_skill = self.get_skill_gateway_config().get("fallback_skill", "lab-task-search-approach")
        if not isinstance(fallback_skill, str) or not fallback_skill:
            raise SkillRuntimeError("skill_gateway.fallback_skill must be a non-empty string")
        return [fallback_skill]

    def get_max_helper_skills(self) -> int:
        value = self.get_skill_gateway_config().get("max_helper_skills", 2)
        if not isinstance(value, int) or value < 0:
            raise SkillRuntimeError("skill_gateway.max_helper_skills must be a non-negative integer")
        return value

    def build_profile(
        self,
        task_name: str,
        mode: str,
        role: str = "main",
        routed_skill_names: Optional[list[str]] = None,
        extra_skill_names: Optional[list[str]] = None,
        user_task_override: Optional[str] = None,
    ) -> PromptProfile:
        task_cfg = self.manifest["tasks"].get(task_name)
        if not isinstance(task_cfg, dict):
            raise SkillRuntimeError(f"Unknown task in framework manifest: {task_name}")

        always_on_skills = self.get_always_on_skill_names()
        routed_skills = routed_skill_names or self.get_task_fallback_dynamic_skills(task_name)
        extra_mode_skills = self._mode_extra_skills(mode, role)
        extra_runtime_skills = extra_skill_names or []
        self._validate_skill_refs(always_on_skills, f"skill_gateway.always_on_skills[{task_name}]")
        self._validate_skill_refs(routed_skills, f"routed_skill_names[{task_name}.{mode}.{role}]")
        self._validate_skill_refs(extra_runtime_skills, f"extra_skill_names[{task_name}.{mode}.{role}]")

        skill_names = self._dedupe_preserve_order([*always_on_skills, *routed_skills, *extra_mode_skills, *extra_runtime_skills])
        rendered = self._render_system_prompt(skill_names)
        return PromptProfile(
            task=task_name,
            mode=mode,
            role=role,
            user_task=user_task_override or task_cfg["user_task"],
            skill_names=skill_names,
            rendered_system_prompt=rendered,
        )

    def build_freeform_profile(
        self,
        task_text: str,
        mode: str,
        role: str = "main",
        routed_skill_names: Optional[list[str]] = None,
        extra_skill_names: Optional[list[str]] = None,
    ) -> PromptProfile:
        clean_task_text = task_text.strip()
        if not clean_task_text:
            raise SkillRuntimeError("free_task text must be non-empty")

        always_on_skills = self.get_always_on_skill_names()
        self._validate_skill_refs(always_on_skills, "skill_gateway.always_on_skills")
        routed = routed_skill_names or self.get_free_task_fallback_dynamic_skills()
        self._validate_skill_refs(routed, "free_task.routed_skill_names")

        extra_mode_skills = self._mode_extra_skills(mode, role)
        extra_runtime_skills = extra_skill_names or []
        self._validate_skill_refs(extra_runtime_skills, "free_task.extra_skill_names")

        skill_names = self._dedupe_preserve_order([*always_on_skills, *routed, *extra_mode_skills, *extra_runtime_skills])
        rendered = self._render_system_prompt(skill_names)
        return PromptProfile(
            task="freeform",
            mode=mode,
            role=role,
            user_task=clean_task_text,
            skill_names=skill_names,
            rendered_system_prompt=rendered,
        )

    def _load_skills(self) -> dict[str, Skill]:
        skills: dict[str, Skill] = {}

        for index, skill_root in enumerate(self.skill_roots):
            if not skill_root.exists():
                if index == 0:
                    raise SkillRuntimeError(f"Skill root does not exist: {skill_root}")
                continue

            for entry in sorted(skill_root.iterdir(), key=lambda item: item.name):
                if not entry.is_dir():
                    continue

                skill_file = entry / "SKILL.md"
                if not skill_file.exists():
                    raise SkillRuntimeError(f"Skill directory missing SKILL.md: {entry}")

                text = skill_file.read_text(encoding="utf-8")
                frontmatter_text, body = self._split_frontmatter(text, skill_file)
                meta = yaml.safe_load(frontmatter_text)
                if not isinstance(meta, dict):
                    raise SkillRuntimeError(f"SKILL.md frontmatter must be a mapping: {skill_file}")

                name = meta.get("name")
                description = meta.get("description")
                self._validate_name(name, entry.name, skill_file)
                self._validate_description(description, skill_file)

                license_value = meta.get("license")
                compatibility = meta.get("compatibility")
                metadata = meta.get("metadata")
                allowed_tools = meta.get("allowed-tools")

                if compatibility is not None:
                    if not isinstance(compatibility, str) or not (1 <= len(compatibility) <= 500):
                        raise SkillRuntimeError(f"Invalid compatibility in {skill_file}")
                if metadata is not None and not isinstance(metadata, dict):
                    raise SkillRuntimeError(f"metadata must be a mapping in {skill_file}")
                if allowed_tools is not None and not isinstance(allowed_tools, str):
                    raise SkillRuntimeError(f"allowed-tools must be a string in {skill_file}")

                if name in skills:
                    continue

                skills[name] = Skill(
                    name=name,
                    description=description.strip(),
                    body=body.strip(),
                    path=skill_file,
                    source_root=skill_root,
                    license=license_value,
                    compatibility=compatibility,
                    metadata=metadata or {},
                    allowed_tools=allowed_tools,
                    frontmatter=meta,
                    body_sections=_parse_markdown_sections(body.strip()),
                )

        if not skills:
            joined = ", ".join(str(root) for root in self.skill_roots)
            raise SkillRuntimeError(f"No skills found under: {joined}")
        return skills

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise SkillRuntimeError(f"Framework manifest does not exist: {self.manifest_path}")
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SkillRuntimeError(f"Invalid JSON in framework manifest: {self.manifest_path}") from exc
        if not isinstance(manifest, dict):
            raise SkillRuntimeError(f"Framework manifest must be a JSON object: {self.manifest_path}")
        return manifest

    def _validate_manifest(self):
        tasks = self.manifest.get("tasks")
        modes = self.manifest.get("modes")
        if not isinstance(tasks, dict):
            raise SkillRuntimeError("framework_manifest.json must contain a 'tasks' object")
        if not isinstance(modes, dict):
            raise SkillRuntimeError("framework_manifest.json must contain a 'modes' object")

        for task_name, task_cfg in tasks.items():
            if not isinstance(task_cfg, dict):
                raise SkillRuntimeError(f"Manifest task '{task_name}' must be an object")
            user_task = task_cfg.get("user_task")
            fallback_dynamic_skills = task_cfg.get("fallback_dynamic_skills")
            if not isinstance(user_task, str) or not user_task.strip():
                raise SkillRuntimeError(f"Manifest task '{task_name}' is missing non-empty user_task")
            if not isinstance(fallback_dynamic_skills, list) or not all(isinstance(name, str) for name in fallback_dynamic_skills):
                raise SkillRuntimeError(f"Manifest task '{task_name}' has invalid fallback_dynamic_skills")
            self._validate_skill_refs(fallback_dynamic_skills, f"tasks.{task_name}.fallback_dynamic_skills")

        expected_mode_shapes = {
            "standard": ("main_extra_skills",),
            "react": ("main_extra_skills",),
            "world_model": ("planner_extra_skills", "selector_extra_skills"),
        }
        for mode_name, keys in expected_mode_shapes.items():
            mode_cfg = modes.get(mode_name)
            if not isinstance(mode_cfg, dict):
                raise SkillRuntimeError(f"Manifest mode '{mode_name}' must be an object")
            for key in keys:
                value = mode_cfg.get(key)
                if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
                    raise SkillRuntimeError(f"Manifest mode '{mode_name}.{key}' must be a list of strings")
                self._validate_skill_refs(value, f"modes.{mode_name}.{key}")

        free_task_cfg = self.get_free_task_config()
        default_tool_profile = free_task_cfg.get("default_tool_profile")
        if not isinstance(default_tool_profile, str) or not default_tool_profile.strip():
            raise SkillRuntimeError("Manifest free_task.default_tool_profile must be a non-empty string")

        gateway_cfg = self.get_skill_gateway_config()
        always_on_skills = gateway_cfg.get("always_on_skills")
        candidate_skills = gateway_cfg.get("candidate_skills")
        fallback_skill = gateway_cfg.get("fallback_skill")
        max_helper_skills = gateway_cfg.get("max_helper_skills")

        if not isinstance(always_on_skills, list) or not all(isinstance(name, str) for name in always_on_skills):
            raise SkillRuntimeError("Manifest skill_gateway.always_on_skills must be a list of strings")
        if not isinstance(candidate_skills, list) or not all(isinstance(name, str) for name in candidate_skills):
            raise SkillRuntimeError("Manifest skill_gateway.candidate_skills must be a list of strings")
        if not isinstance(fallback_skill, str) or not fallback_skill:
            raise SkillRuntimeError("Manifest skill_gateway.fallback_skill must be a non-empty string")
        if not isinstance(max_helper_skills, int) or max_helper_skills < 0:
            raise SkillRuntimeError("Manifest skill_gateway.max_helper_skills must be a non-negative integer")

        self._validate_skill_refs(always_on_skills, "skill_gateway.always_on_skills")
        self._validate_skill_refs(candidate_skills, "skill_gateway.candidate_skills")
        self._validate_skill_refs([fallback_skill], "skill_gateway.fallback_skill")

        overlap = set(always_on_skills) & set(candidate_skills)
        if overlap:
            raise SkillRuntimeError(f"always_on_skills and candidate_skills must not overlap: {', '.join(sorted(overlap))}")

        for skill_name in always_on_skills:
            skill = self.skills[skill_name]
            if skill.skill_kind and skill.skill_kind != "core":
                raise SkillRuntimeError(f"Always-on skill '{skill_name}' must use metadata.skill_kind=core")

        for skill_name in candidate_skills:
            skill = self.skills[skill_name]
            if skill.skill_kind not in {"task", "helper"}:
                raise SkillRuntimeError(
                    f"Gateway candidate skill '{skill_name}' must define metadata.skill_kind as 'task' or 'helper'"
                )
            if not skill.routing_summary.strip():
                raise SkillRuntimeError(f"Gateway candidate skill '{skill_name}' must define metadata.routing_summary")
            if not skill.routing_keywords:
                raise SkillRuntimeError(f"Gateway candidate skill '{skill_name}' must define metadata.routing_keywords")

    def _mode_extra_skills(self, mode: str, role: str) -> list[str]:
        mode_cfg = self.manifest["modes"].get(mode)
        if not isinstance(mode_cfg, dict):
            raise SkillRuntimeError(f"Unknown mode in framework manifest: {mode}")

        if mode in {"standard", "react"}:
            if role != "main":
                raise SkillRuntimeError(f"Mode '{mode}' only supports role 'main', got '{role}'")
            return list(mode_cfg.get("main_extra_skills", []))
        if mode == "world_model":
            key = "planner_extra_skills" if role == "planner" else "selector_extra_skills" if role == "selector" else None
            if key is None:
                raise SkillRuntimeError(f"Mode 'world_model' does not support role '{role}'")
            return list(mode_cfg.get(key, []))
        raise SkillRuntimeError(f"Unknown mode in framework manifest: {mode}")

    def _render_system_prompt(self, skill_names: list[str]) -> str:
        blocks = [
            "You are operating inside SpaceMind with activated Agent Skills.",
            "Everything below is part of your system instructions and must be obeyed.",
            "Activated skills: " + ", ".join(skill_names),
        ]
        for skill_name in skill_names:
            skill = self.skills[skill_name]
            blocks.append(f"===== BEGIN SKILL {skill.name} =====")
            blocks.append(skill.render_block())
            blocks.append(f"===== END SKILL {skill.name} =====")
        return "\n\n".join(blocks).strip()

    def _validate_skill_refs(self, skill_names: list[str], location: str):
        missing = [name for name in skill_names if name not in self.skills]
        if missing:
            raise SkillRuntimeError(f"Unknown skills referenced in {location}: {', '.join(missing)}")

    def _split_frontmatter(self, text: str, skill_file: Path) -> tuple[str, str]:
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            raise SkillRuntimeError(f"SKILL.md must start with YAML frontmatter: {skill_file}")

        end_index = None
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                end_index = idx
                break
        if end_index is None:
            raise SkillRuntimeError(f"SKILL.md frontmatter is not closed with '---': {skill_file}")

        frontmatter = "\n".join(lines[1:end_index])
        body = "\n".join(lines[end_index + 1 :]).strip()
        return frontmatter, body

    def _validate_name(self, name: Any, dir_name: str, skill_file: Path):
        if not isinstance(name, str):
            raise SkillRuntimeError(f"Skill name is required in {skill_file}")
        if not (1 <= len(name) <= 64):
            raise SkillRuntimeError(f"Skill name must be 1-64 chars in {skill_file}")
        if not SKILL_NAME_RE.fullmatch(name):
            raise SkillRuntimeError(f"Skill name has invalid format in {skill_file}: {name}")
        if name != dir_name:
            raise SkillRuntimeError(f"Skill name must match parent directory in {skill_file}: {name} != {dir_name}")

    def _validate_description(self, description: Any, skill_file: Path):
        if not isinstance(description, str) or not description.strip():
            raise SkillRuntimeError(f"Skill description is required in {skill_file}")
        if len(description) > 1024:
            raise SkillRuntimeError(f"Skill description too long in {skill_file}")

    def _dedupe_preserve_order(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered
