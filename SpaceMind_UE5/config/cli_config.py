# 运行方式:
# python host.py --task rendezvous-hold-front --tool_profile hybrid_nav
# python host.py --task search-then-approach --tool_profile vision_only
# python host.py --task inspection-diagnosis --tool_profile hybrid_nav_with_code --enable_react
# python host.py --free_task "Reach a stable point about 2m in front of the target and stop."

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .paths import FRAMEWORK_MANIFEST_PATH, PROJECT_ROOT, ROOT_DOTENV_PATH

load_dotenv(ROOT_DOTENV_PATH)


def _parse_env_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _deep_merge_dict(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_api_keys(api_key_env: str) -> list[str]:
    keys: list[str] = []
    single = (os.getenv(api_key_env) or "").strip()
    if single:
        keys.append(single)

    plural = f"{api_key_env}S"
    raw_multi = os.getenv(plural) or ""
    for part in raw_multi.split(","):
        key = part.strip()
        if key and key not in keys:
            keys.append(key)
    return keys


def _read_api_key_labels(api_key_env: str) -> list[str]:
    raw = os.getenv(f"{api_key_env}_LABELS") or ""
    return [part.strip() for part in raw.split(",") if part.strip()]


def _model(name, max_tokens, temperature, api_key_env, base_url_env, top_p=0.2, extra_body=None):
    return {
        "name": name,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "tool_choice": "auto",
        "extra_body": extra_body or {},
        "api_key": os.getenv(api_key_env),
        "api_keys": _read_api_keys(api_key_env),
        "api_key_labels": _read_api_key_labels(api_key_env),
        "base_url": os.getenv(base_url_env),
    }


def _build_available_models():
    meta = ("METACHAT_API_KEY", "METACHAT_BASE_URL")
    modelscope = ("MODELSCOPE_API_KEY", "MODELSCOPE_BASE_URL")
    return {
        "gpt-4.1": _model("gpt-4.1", 500, 0.3, *meta),
        "claude-sonnet-4": _model("claude-sonnet-4-20250514", 1000, 0.2, *meta),
        "gpt-4o": _model("gpt-4o", 500, 0.3, *meta),
        "qwen3-vl-235b": _model("Qwen/Qwen3-VL-235B-A22B-Instruct:DashScope", 1024, 0.2, *modelscope),
        "qwen3-vl-8b": _model("Qwen/Qwen3-VL-8B-Instruct", 1024, 0.2, *modelscope),
        "qwen3.5-397b": _model(
            "Qwen/Qwen3.5-397B-A17B",
            1024,
            0.6,
            *modelscope,
            top_p=0.95,
            extra_body={"top_k": 20},
        ),
        "qwen3.5-27b": _model("Qwen/Qwen3.5-27B", 1024, 0.2, *modelscope),
        "kimi-k2.5": _model(
            "moonshotai/Kimi-K2.5",
            1024,
            0.6,
            *modelscope,
            top_p=0.95,
        ),
    }


AVAILABLE_MODELS = _build_available_models()
FRAMEWORK_MANIFEST = _load_json(FRAMEWORK_MANIFEST_PATH)
TASK_REGISTRY = FRAMEWORK_MANIFEST.get("tasks", {})
TOOL_PROFILES = FRAMEWORK_MANIFEST.get("tool_profiles", {})
FREE_TASK_CONFIG = FRAMEWORK_MANIFEST.get("free_task", {})
TASK_CHOICES = tuple[Any, ...](TASK_REGISTRY.keys())
TOOL_PROFILE_CHOICES = tuple(TOOL_PROFILES.keys())
DEFAULT_TASK = "rendezvous-hold-front"
DEFAULT_MODEL_ALIAS = "qwen3-vl-235b"
MODEL_ENV_KEY = "SPACEMIND_MODEL"
THINKING_ENV_KEY = "SPACEMIND_THINKING_ENABLED"


class Config:
    task = DEFAULT_TASK
    task_kind = "predefined"
    task_family = "relative-hold"
    free_task = ""
    model = DEFAULT_MODEL_ALIAS
    memory_steps = 8
    server = ""
    tool_profile = ""
    target_name = ""
    no_memory = False
    enable_react = False
    enable_world_model = False
    noise = False
    enable_tta = False
    tta_workspace = "tta_workspace"

    @classmethod
    def get_task_choices(cls) -> tuple[str, ...]:
        return TASK_CHOICES

    @classmethod
    def get_tool_profile_choices(cls) -> tuple[str, ...]:
        return TOOL_PROFILE_CHOICES

    @classmethod
    def get_manifest(cls) -> dict:
        return FRAMEWORK_MANIFEST

    @classmethod
    def get_free_task_config(cls) -> dict:
        if not isinstance(FREE_TASK_CONFIG, dict):
            raise ValueError("Invalid free_task config in framework_manifest.json")
        return FREE_TASK_CONFIG

    @classmethod
    def get_task_config(cls, task_name: str | None = None) -> dict:
        name = task_name or cls.task
        task_cfg = TASK_REGISTRY.get(name)
        if not isinstance(task_cfg, dict):
            raise ValueError(f"Unknown task config: {name}")
        return task_cfg

    @classmethod
    def get_active_task_config(cls) -> dict:
        if cls.is_free_task():
            return {
                "task_name": "freeform",
                "task_family": "freeform",
                "default_tool_profile": cls.get_free_task_config().get("default_tool_profile", "hybrid_nav"),
                "user_task": cls.free_task,
                "fallback_dynamic_skills": [],
            }
        return cls.get_task_config()

    @classmethod
    def get_tool_profile_config(cls, profile_name: str | None = None) -> dict:
        name = profile_name or cls.tool_profile
        profile = TOOL_PROFILES.get(name)
        if not isinstance(profile, dict):
            raise ValueError(f"Unknown tool profile: {name}")
        return profile

    @classmethod
    def is_free_task(cls) -> bool:
        return cls.task_kind == "freeform"

    @classmethod
    def get_runtime_task_name(cls) -> str:
        return "freeform" if cls.is_free_task() else cls.task

    @classmethod
    def parse_args(cls, args=None):
        parser = argparse.ArgumentParser()
        task_group = parser.add_mutually_exclusive_group()
        task_group.add_argument("--task", choices=TASK_CHOICES, default=None)
        task_group.add_argument("--free_task", default="")
        parser.add_argument("--model", default=cls.get_default_model_alias())
        parser.add_argument("--memory_steps", type=int, default=8)
        parser.add_argument("--server", default="")
        parser.add_argument("--tool_profile", choices=TOOL_PROFILE_CHOICES, default=None)
        parser.add_argument("--target_name", default="")
        parser.add_argument("--no_memory", action="store_true")
        parser.add_argument("--enable_react", action="store_true")
        parser.add_argument("--enable_world_model", action="store_true")
        parser.add_argument("--noise", action="store_true")
        parser.add_argument("--enable_tta", action="store_true")
        parser.add_argument("--tta_workspace", default="tta_workspace")
        parsed = parser.parse_args(args)

        raw_free_task = parsed.free_task if isinstance(parsed.free_task, str) else ""
        free_task_text = raw_free_task.strip()
        selected_task = parsed.task or DEFAULT_TASK

        if raw_free_task and not free_task_text:
            parser.error("--free_task 不能为空白文本。")

        cls.model = parsed.model
        cls.memory_steps = parsed.memory_steps
        cls.server = parsed.server
        cls.target_name = parsed.target_name.strip()
        cls.no_memory = parsed.no_memory
        cls.enable_react = parsed.enable_react
        cls.enable_world_model = parsed.enable_world_model
        cls.noise = parsed.noise
        cls.enable_tta = parsed.enable_tta

        if free_task_text:
            cls.task_kind = "freeform"
            cls.task = "freeform"
            cls.task_family = "freeform"
            cls.free_task = free_task_text
            default_profile = str(cls.get_free_task_config().get("default_tool_profile", "hybrid_nav"))
            cls.tool_profile = parsed.tool_profile or default_profile
            cls.tta_workspace = parsed.tta_workspace
            if cls.enable_tta:
                parser.error("--free_task 暂不支持 TTA，请移除 --enable_tta。")
        else:
            cls.task_kind = "predefined"
            cls.task = selected_task
            cls.free_task = ""
            task_cfg = cls.get_task_config(selected_task)
            cls.task_family = str(task_cfg.get("task_family", selected_task))
            cls.tool_profile = parsed.tool_profile or str(task_cfg.get("default_tool_profile", "hybrid_nav"))
            cls.tta_workspace = str(Path("tta_workspace") / cls.task) if parsed.tta_workspace == "tta_workspace" else parsed.tta_workspace

        return parsed

    @classmethod
    def get_default_model_alias(cls) -> str:
        env_model = (os.getenv(MODEL_ENV_KEY) or "").strip()
        if env_model in AVAILABLE_MODELS:
            return env_model
        return DEFAULT_MODEL_ALIAS

    @classmethod
    def get_thinking_enabled(cls) -> bool:
        return _parse_env_bool(os.getenv(THINKING_ENV_KEY), default=True)

    @classmethod
    def _thinking_extra_body(cls, model_alias: str, thinking_enabled: bool) -> dict | None:
        if model_alias == "kimi-k2.5":
            if thinking_enabled:
                return {}
            return {"chat_template_kwargs": {"thinking": False}}

        if model_alias in {"qwen3.5-397b", "qwen3.5-27b"}:
            return {"chat_template_kwargs": {"enable_thinking": thinking_enabled}}

        return None

    @classmethod
    def get_model_config(cls):
        model_alias = cls.model if cls.model in AVAILABLE_MODELS else DEFAULT_MODEL_ALIAS
        config = copy.deepcopy(AVAILABLE_MODELS[model_alias])
        thinking_enabled = cls.get_thinking_enabled()
        thinking_override = cls._thinking_extra_body(model_alias, thinking_enabled)

        config["alias"] = model_alias
        config["thinking_requested"] = thinking_enabled
        config["thinking_supported"] = thinking_override is not None

        if thinking_override is not None:
            config["extra_body"] = _deep_merge_dict(config.get("extra_body", {}), thinking_override)
            config["thinking_effective"] = thinking_enabled
        else:
            config["thinking_effective"] = None

        return config

    @classmethod
    def get_runtime_mode(cls) -> str:
        if cls.enable_world_model:
            return "world_model"
        if cls.enable_react:
            return "react"
        return "standard"

    @classmethod
    def show_info(cls):
        mc = cls.get_model_config()
        tool_profile = cls.get_tool_profile_config()
        print(f"task={cls.get_runtime_task_name()} family={cls.task_family} model={cls.model}")
        if cls.is_free_task():
            print(f"free_task={cls.free_task}")
        print(f"model: {mc['name']} max_tokens={mc['max_tokens']} temperature={mc['temperature']} top_p={mc['top_p']}")
        print(f"tool_profile={cls.tool_profile} intended_use={tool_profile.get('intended_use', '')} target_name={cls.target_name or 'default'}")
        print(f"code_execution_guidance={cls.tool_profile == 'hybrid_nav_with_code'}  # profile-controlled")
        print(
            f"thinking: requested={mc['thinking_requested']} supported={mc['thinking_supported']}"
            + (f" effective={mc['thinking_effective']}" if mc["thinking_supported"] else " effective=no-op")
        )
        print(f"ablation: no_memory={cls.no_memory}")
        print(f"ablation: enable_react={cls.enable_react} enable_world_model={cls.enable_world_model} noise={cls.noise}")
        print(f"tta: enable_tta={cls.enable_tta} workspace={cls.tta_workspace}")
