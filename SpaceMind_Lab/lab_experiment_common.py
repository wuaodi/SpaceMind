#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from runtime_logs.result_parser import (
    build_benchmark_summary,
    export_benchmark_csv,
    export_csv,
    parse_single_log,
)

PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = PROJECT_ROOT / "runtime_logs" / "benchmark"
LOG_DIR = PROJECT_ROOT / "runtime_logs" / "log"
ANNOTATION_TEMPLATE_PATH = PROJECT_ROOT / "config" / "evaluation" / "benchmark_annotation_template.json"
HOST_SCRIPT = PROJECT_ROOT / "host.py"
DEFAULT_TOOL_PROFILE = "lab_nav_minimal"
DEFAULT_MODE = "standard"
DEFAULT_TTA_ROUNDS = 5
TARGET_DISTANCE_M = 1.0

SATELLITES = list(
    json.loads(ANNOTATION_TEMPLATE_PATH.read_text(encoding="utf-8")).get("satellites", {}).keys()
)

TASKS = [
    "rendezvous-hold-front",
    "search-then-approach",
    "inspection-diagnosis",
]

NAVIGATION_TASKS = {"rendezvous-hold-front", "search-then-approach"}

MODE_FLAGS: dict[str, list[str]] = {
    "standard": [],
    "react": ["--enable_react"],
    "world_model": ["--enable_world_model"],
}

PHASE_A_CONDITION_HINTS = {
    "rendezvous-hold-front": {
        "C1": "目标可见，较标准正前方，距离较近。",
        "C2": "目标可见，更远且带偏角。",
    },
    "search-then-approach": {
        "C1": "目标初始不可见，偏右侧。",
        "C2": "目标初始不可见，偏左侧且更远。",
    },
    "inspection-diagnosis": {
        "C1": "目标可见，正前方标准视角。",
        "C2": "目标可见，斜视角。",
    },
}


def ensure_benchmark_dir() -> Path:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    return BENCHMARK_DIR


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def prompt_text(label: str, default: str = "", required: bool = True) -> str:
    while True:
        suffix = f" [{default}]" if default else ""
        raw = input(f"{label}{suffix}: ").strip()
        value = raw or default
        if value or not required:
            return value
        print("输入不能为空，请重新输入。")


def prompt_float(label: str, default: float | None = None, required: bool = True) -> float | None:
    while True:
        shown_default = "" if default is None else str(default)
        text = prompt_text(label, shown_default, required=required)
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            print("请输入数字，例如 2.0 或 -15。")


def prompt_bool(label: str, default: bool | None = None) -> bool:
    default_text = ""
    if default is True:
        default_text = "y"
    elif default is False:
        default_text = "n"

    while True:
        raw = prompt_text(f"{label} (y/n)", default_text, required=default is None).lower()
        if raw in {"y", "yes", "true", "1"}:
            return True
        if raw in {"n", "no", "false", "0"}:
            return False
        print("请输入 y 或 n。")


def wait_for_operator(message: str) -> None:
    input(f"{message}\n按回车继续...")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_records_csv_jsonl(csv_path: Path, jsonl_path: Path, records: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if records:
        export_csv(records, str(csv_path))
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def bool_to_str(value: bool) -> str:
    return "True" if value else "False"


def str_to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def nullable_float_to_str(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def parse_float(value: Any) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def snapshot_logs() -> set[str]:
    return {path.name for path in LOG_DIR.glob("spacemind_*.txt")}


def detect_new_log(existing_logs: set[str]) -> Path | None:
    candidates = [path for path in LOG_DIR.glob("spacemind_*.txt") if path.name not in existing_logs]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_host(
    *,
    task: str,
    satellite: str,
    mode: str,
    model: str = "",
    enable_tta: bool = False,
    tta_workspace: str = "",
) -> tuple[int, Path | None, dict[str, Any]]:
    existing_logs = snapshot_logs()
    cmd = [
        sys.executable,
        str(HOST_SCRIPT),
        "--task",
        task,
        "--target_name",
        satellite,
        "--tool_profile",
        DEFAULT_TOOL_PROFILE,
    ]
    if model:
        cmd.extend(["--model", model])
    cmd.extend(MODE_FLAGS[mode])
    if enable_tta:
        cmd.append("--enable_tta")
        if tta_workspace:
            cmd.extend(["--tta_workspace", tta_workspace])

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    log_path = detect_new_log(existing_logs)
    auto_record = parse_single_log(str(log_path)) if log_path else {}
    return result.returncode, log_path, auto_record


def condition_hint(task: str, condition_id: str) -> str:
    return PHASE_A_CONDITION_HINTS.get(task, {}).get(condition_id, "")


def collect_initial_metadata(
    *,
    phase: str,
    satellite: str,
    task: str,
    mode: str,
    condition_id: str,
    tta_round: int = 0,
) -> dict[str, str]:
    print(f"\n[{phase}] 当前运行: satellite={satellite} task={task} mode={mode} condition={condition_id}")
    hint = condition_hint(task, condition_id)
    if hint:
        print(f"建议条件说明: {hint}")
    return {
        "satellite": prompt_text("satellite", satellite),
        "task": prompt_text("task", task),
        "mode": prompt_text("mode", mode),
        "condition_id": prompt_text("condition_id", condition_id),
        "initial_distance_m": nullable_float_to_str(prompt_float("initial_distance_m", required=True)),
        "initial_relative_angle_deg": nullable_float_to_str(prompt_float("initial_relative_angle_deg", required=True)),
        "target_visible_at_start": bool_to_str(prompt_bool("target_visible_at_start", default=(task != "search-then-approach"))),
        "initial_notes": prompt_text("initial_notes", "", required=False),
        "tta_round": str(tta_round),
    }


def collect_final_metadata(*, task: str) -> dict[str, str]:
    nav_task = task in NAVIGATION_TASKS
    final_distance = prompt_float(
        "final_measured_distance_m" + ("" if nav_task else "（inspection 可留空）"),
        required=nav_task,
    )
    return {
        "final_measured_distance_m": nullable_float_to_str(final_distance),
        "collision_observed": bool_to_str(prompt_bool("collision_observed", default=False)),
        "manual_success": bool_to_str(prompt_bool("manual_success", default=False)),
        "manual_failure_reason": prompt_text("manual_failure_reason", "", required=False),
        "final_notes": prompt_text("final_notes", "", required=False),
    }


def build_manual_record(
    *,
    phase: str,
    run_index: int,
    satellite: str,
    task: str,
    mode: str,
    condition_id: str,
    log_file: str,
    host_returncode: int,
    initial_meta: dict[str, str],
    final_meta: dict[str, str],
    enable_tta: bool = False,
    tta_workspace: str = "",
    tta_round: int = 0,
) -> dict[str, Any]:
    record = {
        "phase": phase,
        "run_index": run_index,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "satellite": satellite,
        "task": task,
        "mode": mode,
        "condition_id": condition_id,
        "tool_profile": DEFAULT_TOOL_PROFILE,
        "enable_tta": bool_to_str(enable_tta),
        "tta_workspace": tta_workspace,
        "tta_round": tta_round,
        "log_file": log_file,
        "host_returncode": host_returncode,
    }
    record.update(initial_meta)
    record.update(final_meta)
    return record


def build_auto_record(
    *,
    phase: str,
    satellite: str,
    task: str,
    mode: str,
    condition_id: str,
    log_file: str,
    host_returncode: int,
    auto_record: dict[str, Any],
    enable_tta: bool = False,
    tta_workspace: str = "",
) -> dict[str, Any]:
    record = dict(auto_record or {})
    record.update(
        {
            "phase": phase,
            "satellite": satellite,
            "task": task,
            "mode": mode,
            "condition_id": condition_id,
            "tool_profile": DEFAULT_TOOL_PROFILE,
            "enable_tta": bool_to_str(enable_tta),
            "tta_workspace": tta_workspace,
            "log_file": log_file,
            "host_returncode": host_returncode,
        }
    )
    return record


def join_manual_auto(manual_records: list[dict[str, Any]], auto_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    auto_by_log = {str(item.get("log_file", "")): item for item in auto_records}
    joined = []
    for manual in manual_records:
        row = dict(manual)
        row.update(auto_by_log.get(str(manual.get("log_file", "")), {}))
        joined.append(row)
    return joined


def build_phase_a_mode_summary(
    manual_records: list[dict[str, Any]],
    auto_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    joined = join_manual_auto(manual_records, auto_records)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        grouped[(str(row.get("task", "")), str(row.get("mode", "")))].append(row)

    summary = []
    for (task, mode), items in sorted(grouped.items()):
        success_values = [1.0 if str_to_bool(item.get("manual_success")) else 0.0 for item in items]
        collision_values = [1.0 if str_to_bool(item.get("collision_observed")) else 0.0 for item in items]
        manual_distance_values = [value for item in items if (value := parse_float(item.get("final_measured_distance_m"))) is not None]
        auto_distance_values = [value for item in items if (value := parse_float(item.get("surface_distance_m"))) is not None]
        inspection_scores = [value for item in items if (value := parse_float(item.get("inspection_score"))) is not None]
        step_values = [value for item in items if (value := parse_float(item.get("total_steps"))) is not None]
        summary.append(
            {
                "task": task,
                "mode": mode,
                "runs": len(items),
                "manual_success_rate": mean(success_values),
                "manual_collision_rate": mean(collision_values),
                "avg_manual_distance_m": mean(manual_distance_values),
                "avg_auto_surface_distance_m": mean(auto_distance_values),
                "avg_inspection_score": mean(inspection_scores),
                "avg_steps": mean(step_values),
            }
        )
    return summary


def rank_navigation_combos(
    manual_records: list[dict[str, Any]],
    auto_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    joined = [row for row in join_manual_auto(manual_records, auto_records) if str(row.get("task")) in NAVIGATION_TASKS]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        grouped[(str(row.get("satellite", "")), str(row.get("task", "")), str(row.get("mode", "")))].append(row)

    ranked = []
    for (satellite, task, mode), items in sorted(grouped.items()):
        condition_rows = sorted(items, key=single_run_worse_key)
        distance_errors = []
        steps = []
        collisions = []
        successes = []
        for item in items:
            collisions.append(1.0 if str_to_bool(item.get("collision_observed")) else 0.0)
            successes.append(1.0 if str_to_bool(item.get("manual_success")) else 0.0)
            if (distance := parse_float(item.get("final_measured_distance_m"))) is not None:
                distance_errors.append(abs(distance - TARGET_DISTANCE_M))
            if (step := parse_float(item.get("total_steps"))) is not None:
                steps.append(step)

        ranked.append(
            {
                "satellite": satellite,
                "task": task,
                "mode": mode,
                "runs": len(items),
                "success_rate": mean(successes),
                "collision_rate": mean(collisions),
                "avg_distance_error_m": mean(distance_errors),
                "avg_steps": mean(steps),
                "worst_condition_id": condition_rows[0].get("condition_id", "") if condition_rows else "",
                "worst_log_file": condition_rows[0].get("log_file", "") if condition_rows else "",
            }
        )

    ranked.sort(
        key=lambda item: (
            item["success_rate"],
            -item["collision_rate"],
            -item["avg_distance_error_m"],
            -item["avg_steps"],
            item["satellite"],
            item["task"],
            item["mode"],
        )
    )
    return ranked


def single_run_worse_key(item: dict[str, Any]) -> tuple[float, float, float, float, str]:
    success_value = 1.0 if str_to_bool(item.get("manual_success")) else 0.0
    collision_value = 1.0 if str_to_bool(item.get("collision_observed")) else 0.0
    distance = parse_float(item.get("final_measured_distance_m"))
    distance_error = abs(distance - TARGET_DISTANCE_M) if distance is not None else -1.0
    steps = parse_float(item.get("total_steps")) or 0.0
    return (
        success_value,
        -collision_value,
        -distance_error,
        -steps,
        str(item.get("condition_id", "")),
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def export_benchmark_rows(auto_records: list[dict[str, Any]], output_csv: Path) -> None:
    if not auto_records:
        return
    rows = build_benchmark_summary(auto_records)
    if rows:
        export_benchmark_csv(rows, str(output_csv))


def build_phase_b_tta_summary(
    manual_records: list[dict[str, Any]],
    auto_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    joined = sorted(
        join_manual_auto(manual_records, auto_records),
        key=lambda row: int(str(row.get("tta_round", "0")) or 0),
    )
    if not joined:
        return []

    first = joined[0]
    last = joined[-1]
    success_values = [1.0 if str_to_bool(item.get("manual_success")) else 0.0 for item in joined]
    distance_values = [value for item in joined if (value := parse_float(item.get("final_measured_distance_m"))) is not None]
    steps = [value for item in joined if (value := parse_float(item.get("total_steps"))) is not None]
    return [
        {
            "satellite": first.get("satellite", ""),
            "task": first.get("task", ""),
            "mode": first.get("mode", ""),
            "condition_id": first.get("condition_id", ""),
            "tta_workspace": first.get("tta_workspace", ""),
            "runs": len(joined),
            "success_rate": mean(success_values),
            "first_round_success": first.get("manual_success", ""),
            "last_round_success": last.get("manual_success", ""),
            "first_round_distance_m": first.get("final_measured_distance_m", ""),
            "last_round_distance_m": last.get("final_measured_distance_m", ""),
            "avg_distance_m": mean(distance_values),
            "avg_steps": mean(steps),
            "first_round_termination": first.get("failure_mode", ""),
            "last_round_termination": last.get("failure_mode", ""),
            "learning_signal": learning_signal(first, last),
        }
    ]


def learning_signal(first: dict[str, Any], last: dict[str, Any]) -> str:
    first_success = str_to_bool(first.get("manual_success"))
    last_success = str_to_bool(last.get("manual_success"))
    if not first_success and last_success:
        return "improved_to_success"
    if first_success and not last_success:
        return "regressed"
    if first_success and last_success:
        return "stable_success"
    return "no_clear_gain"
