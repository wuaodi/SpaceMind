import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from config.paths import FRAMEWORK_MANIFEST_PATH, RUNTIME_LOG_DIR


def parse_single_log(filepath: str) -> dict:
    path = Path(filepath)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    result = {
        "log_file": path.name,
        "model": "",
        "task_type": "unknown",
        "task_kind": "predefined",
        "free_task_text": "",
        "success": False,
        "allow_code_exec": False,
        "code_execution_guidance": False,
        "evaluator_success": None,
        "evaluator_profile": "",
        "evaluator_surface_distance_m": None,
        "evaluator_failure_reason": "",
        "inspection_score": None,
        "inspection_scoring_reason": "",
        "total_steps": 0,
        "tool_calls": 0,
        "sensing_calls": 0,
        "move_calls": 0,
        "failure_mode": "unknown",
        "duration_seconds": 0,
        "episode_index": 0,
        "learned_skill_count": 0,
        "applied_skill_count": 0,
        "mutation_type": "",
        "tool_profile": "",
        "task_family": "",
        "target_name": "",
        "collision_flag": 0,
        "overshoot_flag": 0,
        "target_lost_flag": 0,
        "skill_reused": 0,
        "sensing_to_motion_ratio": 0.0,
    }

    step_re = re.compile(r"Step (\d+)")
    tool_re = re.compile(r"Calling tool(?: \([^)]*\))?: (\w+)")
    term_re = re.compile(r"Termination reason: (.+?)(?:\n|$)", re.DOTALL)
    model_re = re.compile(r"Current model: (\S+)")
    task_done_re = re.compile(r"Task completed!")
    mission_success_re = re.compile(r"Mission success: (True|False)")
    allow_code_exec_re = re.compile(r"allow_code_exec=(True|False)")
    code_execution_guidance_re = re.compile(r"code_execution_guidance=(True|False)")
    evaluator_re = re.compile(
        r"Evaluator result: success=(True|False) profile=([^\s]+) surface_distance_m=([^\s]+) "
        r"collision_detected=(True|False) failure_reason=([^\s]+)"
    )
    inspection_evaluator_re = re.compile(
        r"Inspection evaluator result: profile=([^\s]+) target=([^\s]+) score=([^\s]+)"
        r"(?: breakdown=\[[^\]]*\])? reason=(.+)"
    )
    flyaround_detail_re = re.compile(
        r"FlyAround detail: center_distance_m=([^\s]+) angle_from_vertical_deg=([^\s]+)"
    )
    ts_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - ")
    tta_episode_index_re = re.compile(r"TTA episode index: (\d+)")
    tta_learned_count_re = re.compile(r"TTA learned skill count(?: \(after\))?: (\d+)")
    tta_applied_count_re = re.compile(r"TTA applied skill count: (\d+)")
    tta_mutation_type_re = re.compile(r"TTA mutation type: ([\w-]+)")
    benchmark_re = re.compile(r"Benchmark config: task=(.+?) task_family=(.+?) target_name=(.+?) tool_profile=(.+)$")
    task_config_re = re.compile(
        r"Task config: task=(.+?) task_kind=(.+?) task_family=(.+?) target_name=(.+?) tool_profile=(.+)$"
    )
    free_task_re = re.compile(r"Free task text: (.+)$")

    sensing_tools = {
        "lidar_info",
        "image_bright",
        "part_segmentation",
        "knowledge_base",
        "image_zoom",
        "image_crop",
        "set_exposure",
        "component_detection",
        "visibility_score",
    }
    move_tools = {"set_position", "set_attitude"}

    steps = []
    tools_called = []
    first_ts = None
    last_ts = None
    explicit_mission_success = None

    for line in lines:
        ts_m = ts_re.match(line)
        if ts_m:
            try:
                dt = datetime.strptime(ts_m.group(1), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                dt = None
            if dt is not None:
                if first_ts is None:
                    first_ts = dt
                last_ts = dt

        step_m = step_re.search(line)
        if step_m:
            steps.append(int(step_m.group(1)))

        tool_m = tool_re.search(line)
        if tool_m:
            tools_called.append(tool_m.group(1))

        model_m = model_re.search(line)
        if model_m:
            result["model"] = model_m.group(1).strip()

        allow_code_exec_m = allow_code_exec_re.search(line)
        if allow_code_exec_m:
            result["allow_code_exec"] = allow_code_exec_m.group(1) == "True"

        code_execution_guidance_m = code_execution_guidance_re.search(line)
        if code_execution_guidance_m:
            result["code_execution_guidance"] = code_execution_guidance_m.group(1) == "True"

        mission_success_m = mission_success_re.search(line)
        if mission_success_m:
            explicit_mission_success = mission_success_m.group(1) == "True"

        evaluator_m = evaluator_re.search(line)
        if evaluator_m:
            result["evaluator_success"] = evaluator_m.group(1) == "True"
            result["evaluator_profile"] = evaluator_m.group(2)
            surface_text = evaluator_m.group(3)
            result["evaluator_surface_distance_m"] = None if surface_text == "na" else float(surface_text)
            result["evaluator_failure_reason"] = evaluator_m.group(5)

        inspection_evaluator_m = inspection_evaluator_re.search(line)
        if inspection_evaluator_m:
            result["evaluator_profile"] = inspection_evaluator_m.group(1)
            score_text = inspection_evaluator_m.group(3)
            result["inspection_score"] = None if score_text == "na" else float(score_text)
            result["inspection_scoring_reason"] = inspection_evaluator_m.group(4).strip()

        flyaround_m = flyaround_detail_re.search(line)
        if flyaround_m:
            dist_text, angle_text = flyaround_m.group(1), flyaround_m.group(2)
            result["flyaround_center_distance_m"] = None if dist_text == "na" else float(dist_text)
            result["flyaround_angle_deg"] = None if angle_text == "na" else float(angle_text)

        tta_episode_index_m = tta_episode_index_re.search(line)
        if tta_episode_index_m:
            result["episode_index"] = int(tta_episode_index_m.group(1))

        tta_learned_count_m = tta_learned_count_re.search(line)
        if tta_learned_count_m:
            result["learned_skill_count"] = int(tta_learned_count_m.group(1))

        tta_applied_count_m = tta_applied_count_re.search(line)
        if tta_applied_count_m:
            result["applied_skill_count"] = int(tta_applied_count_m.group(1))

        tta_mutation_type_m = tta_mutation_type_re.search(line)
        if tta_mutation_type_m:
            result["mutation_type"] = tta_mutation_type_m.group(1)

        benchmark_m = benchmark_re.search(line)
        if benchmark_m:
            result["task_type"] = benchmark_m.group(1).strip()
            result["task_family"] = benchmark_m.group(2).strip()
            target_name = benchmark_m.group(3).strip()
            result["target_name"] = "" if target_name == "default" else target_name
            result["tool_profile"] = benchmark_m.group(4).strip()

        task_config_m = task_config_re.search(line)
        if task_config_m:
            result["task_type"] = task_config_m.group(1).strip()
            result["task_kind"] = task_config_m.group(2).strip()
            result["task_family"] = task_config_m.group(3).strip()
            target_name = task_config_m.group(4).strip()
            result["target_name"] = "" if target_name == "default" else target_name
            result["tool_profile"] = task_config_m.group(5).strip()

        free_task_m = free_task_re.search(line)
        if free_task_m:
            result["free_task_text"] = free_task_m.group(1).strip()

    if "_traj1_" in path.name or "_code1_" in path.name:
        result["allow_code_exec"] = True
    if "_hybrid_nav_with_code_" in path.name:
        result["allow_code_exec"] = True

    if result["tool_profile"] == "hybrid_nav_with_code":
        result["allow_code_exec"] = True
    if result["code_execution_guidance"]:
        result["allow_code_exec"] = True

    result["total_steps"] = max(steps) if steps else 0
    result["tool_calls"] = len(tools_called)
    result["sensing_calls"] = sum(1 for tool_name in tools_called if tool_name in sensing_tools)
    result["move_calls"] = sum(1 for tool_name in tools_called if tool_name in move_tools)

    if first_ts and last_ts:
        result["duration_seconds"] = int((last_ts - first_ts).total_seconds())

    full_text = text
    term_m = term_re.search(full_text)
    term_reason = term_m.group(1).strip() if term_m else ""
    if explicit_mission_success is True:
        result["success"] = True
        result["failure_mode"] = "success"
    elif explicit_mission_success is False and term_reason:
        term_lower = term_reason.lower()
        if "target lost" in term_lower:
            result["failure_mode"] = "target_lost"
        elif "collision" in term_lower:
            result["failure_mode"] = "collision"
        elif "timeout" in term_lower:
            result["failure_mode"] = "timeout"
        elif "overshoot" in term_lower:
            result["failure_mode"] = "overshoot"
        else:
            result["failure_mode"] = "early_termination"
    elif term_reason:
        term_lower = term_reason.lower()
        if "task completed" in term_lower or "mission complete" in term_lower or "perception task completed" in term_lower:
            result["success"] = True
            result["failure_mode"] = "success"
        elif "target lost" in term_lower:
            result["failure_mode"] = "target_lost"
        elif "collision" in term_lower:
            result["failure_mode"] = "collision"
        elif "timeout" in term_lower:
            result["failure_mode"] = "timeout"
        elif "overshoot" in term_lower:
            result["failure_mode"] = "overshoot"
        else:
            result["failure_mode"] = "early_termination"
    elif task_done_re.search(full_text):
        result["success"] = True
        result["failure_mode"] = "success"

    if result["evaluator_success"] is not None:
        result["success"] = bool(result["evaluator_success"])
        if result["success"]:
            result["failure_mode"] = "success"
        elif result["evaluator_failure_reason"]:
            result["failure_mode"] = result["evaluator_failure_reason"]

    text_lower = full_text.lower()
    if result["task_type"] == "unknown":
        if "task=rendezvous-hold-front" in text_lower:
            result["task_type"] = "rendezvous-hold-front"
        elif "task=search-then-approach" in text_lower:
            result["task_type"] = "search-then-approach"
        elif "task=inspection-diagnosis" in text_lower:
            result["task_type"] = "inspection-diagnosis"
    if not result["task_family"]:
        result["task_family"] = result["task_type"]

    result["collision_flag"] = int(result["failure_mode"] == "collision")
    result["overshoot_flag"] = int(result["failure_mode"] == "overshoot")
    result["target_lost_flag"] = int(result["failure_mode"] == "target_lost")
    result["skill_reused"] = int(result["applied_skill_count"] > 0)
    if result["move_calls"] > 0:
        result["sensing_to_motion_ratio"] = result["sensing_calls"] / result["move_calls"]

    return result


def parse_all_logs(log_dir: str, patterns: list | None = None) -> list:
    log_path = Path(log_dir)
    if not log_path.is_dir():
        return []
    patterns = patterns or ["spacemind_*.txt", "*_run*.txt", "log*.txt"]
    results = []
    seen = set()
    for pattern in patterns:
        for file_path in log_path.glob(pattern):
            if file_path.name in seen:
                continue
            seen.add(file_path.name)
            parsed = parse_single_log(str(file_path))
            if parsed:
                results.append(parsed)
    return results


def export_csv(results: list, output_path: str):
    if not results:
        return
    keys = list(results[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def load_framework_manifest() -> dict:
    if not FRAMEWORK_MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(FRAMEWORK_MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _window_mean(values: list[float], window: int) -> float:
    if not values:
        return 0.0
    window = max(1, min(window, len(values)))
    return _mean(values[:window])


def _last_window_mean(values: list[float], window: int) -> float:
    if not values:
        return 0.0
    window = max(1, min(window, len(values)))
    return _mean(values[-window:])


def _edge_window_size(length: int, max_window: int = 5) -> int:
    if length <= 0:
        return 1
    if length >= max_window * 2:
        return max_window
    return max(1, length // 2)


def _area_under_learning_curve(success_values: list[float]) -> float:
    if not success_values:
        return 0.0
    prefix_means = []
    running = 0.0
    for idx, value in enumerate(success_values, 1):
        running += value
        prefix_means.append(running / idx)
    return _mean(prefix_means)


def _sorted_for_learning(items: list[dict]) -> list[dict]:
    def sort_key(item: dict) -> tuple[int, str]:
        episode_index = int(item.get("episode_index", 0) or 0)
        return (episode_index if episode_index > 0 else 10**9, str(item.get("log_file", "")))

    return sorted(items, key=sort_key)


def build_benchmark_summary(results: list) -> list[dict]:
    if not results:
        return []

    framework_manifest = load_framework_manifest()
    task_family_cfg = framework_manifest.get("task_families", {}) if isinstance(framework_manifest, dict) else {}

    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for result in results:
        key = (
            result.get("model", ""),
            result.get("task_family", ""),
            result.get("task_type", ""),
            result.get("tool_profile", ""),
        )
        grouped[key].append(result)

    summary_rows = []
    for key, items in sorted(grouped.items()):
        model, task_family, task_type, tool_profile = key
        family_meta = task_family_cfg.get(task_family, {})
        sorted_items = _sorted_for_learning(items)
        success_curve = [1.0 if item.get("success") else 0.0 for item in sorted_items]
        edge_window = _edge_window_size(len(success_curve), 5)
        first5 = _window_mean(success_curve, edge_window)
        last5 = _last_window_mean(success_curve, edge_window)
        first_half = _window_mean(success_curve, max(1, len(success_curve) // 2))
        second_half = _last_window_mean(success_curve, max(1, len(success_curve) // 2))
        mutation_distribution = defaultdict(int)
        for item in items:
            mutation_distribution[str(item.get("mutation_type", "") or "none")] += 1

        row = {
            "model": model,
            "task_family": task_family,
            "task_type": task_type,
            "tool_profile": tool_profile,
            "runs": len(items),
            "success_rate": _mean([1.0 if item.get("success") else 0.0 for item in items]),
            "collision_rate": _mean([float(item.get("collision_flag", 0) or 0) for item in items]),
            "overshoot_rate": _mean([float(item.get("overshoot_flag", 0) or 0) for item in items]),
            "target_lost_rate": _mean([float(item.get("target_lost_flag", 0) or 0) for item in items]),
            "avg_steps": _mean([float(item.get("total_steps", 0) or 0) for item in items]),
            "avg_tool_calls": _mean([float(item.get("tool_calls", 0) or 0) for item in items]),
            "avg_sensing_calls": _mean([float(item.get("sensing_calls", 0) or 0) for item in items]),
            "avg_motion_calls": _mean([float(item.get("move_calls", 0) or 0) for item in items]),
            "avg_duration_seconds": _mean([float(item.get("duration_seconds", 0) or 0) for item in items]),
            "avg_sensing_to_motion_ratio": _mean([float(item.get("sensing_to_motion_ratio", 0.0) or 0.0) for item in items]),
            "avg_surface_distance_m": _mean(
                [
                    float(item["evaluator_surface_distance_m"])
                    for item in items
                    if item.get("evaluator_surface_distance_m") is not None
                ]
            ),
            "avg_inspection_score": _mean(
                [
                    float(item["inspection_score"])
                    for item in items
                    if item.get("inspection_score") is not None
                ]
            ),
            "avg_learned_skill_count": _mean([float(item.get("learned_skill_count", 0) or 0) for item in items]),
            "avg_applied_skill_count": _mean([float(item.get("applied_skill_count", 0) or 0) for item in items]),
            "skill_reuse_rate": _mean([float(item.get("skill_reused", 0) or 0) for item in items]),
            "learning_gain": second_half - first_half,
            "area_under_learning_curve": _area_under_learning_curve(success_curve),
            "last5_minus_first5": last5 - first5,
            "mutation_type_distribution": json.dumps(dict(sorted(mutation_distribution.items())), ensure_ascii=False),
            "task_family_description": family_meta.get("description", ""),
            "primary_metrics": ",".join(family_meta.get("primary_metrics", [])),
        }
        summary_rows.append(row)

    return summary_rows


def export_benchmark_csv(summary_rows: list, output_path: str):
    if not summary_rows:
        return
    keys = list(summary_rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary_rows)


def print_summary(results: list):
    if not results:
        print("No results.")
        return
    by_model = defaultdict(list)
    for result in results:
        by_model[result.get("model", "unknown")].append(result)

    print("\n=== Summary by Model ===")
    for model, items in sorted(by_model.items()):
        total = len(items)
        success = sum(1 for item in items if item.get("success"))
        avg_steps = sum(item.get("total_steps", 0) for item in items) / total if total else 0
        avg_tools = sum(item.get("tool_calls", 0) for item in items) / total if total else 0
        print(f"  {model}: success={success}/{total} ({100 * success / total:.1f}%), avg_steps={avg_steps:.1f}, avg_tool_calls={avg_tools:.1f}")

    print("\n=== Failure Modes ===")
    by_failure = defaultdict(int)
    for result in results:
        by_failure[result.get("failure_mode", "unknown")] += 1
    for mode, count in sorted(by_failure.items(), key=lambda item: -item[1]):
        print(f"  {mode}: {count}")


def print_benchmark_summary(summary_rows: list):
    if not summary_rows:
        print("\n=== Benchmark Summary ===")
        print("No benchmark rows.")
        return

    print("\n=== Benchmark Summary ===")
    for row in summary_rows:
        print(
            "  "
            f"{row['model']} | {row['task_family']} | {row['task_type']} | "
            f"profile={row['tool_profile']} | "
            f"success={row['success_rate']:.3f} | collision={row['collision_rate']:.3f} | "
            f"avg_steps={row['avg_steps']:.1f} | "
            f"surface={row['avg_surface_distance_m']:.3f} | "
            f"inspection_score={row['avg_inspection_score']:.2f} | "
            f"learning_gain={row['learning_gain']:.3f} | reuse={row['skill_reuse_rate']:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default=str(RUNTIME_LOG_DIR))
    parser.add_argument("--output", default="results.csv")
    parser.add_argument("--benchmark_output", default="")
    args = parser.parse_args()
    parsed_results = parse_all_logs(args.log_dir)
    if parsed_results:
        export_csv(parsed_results, args.output)
        print_summary(parsed_results)
        benchmark_rows = build_benchmark_summary(parsed_results)
        if benchmark_rows:
            benchmark_output = args.benchmark_output or str(Path(args.output).with_name(Path(args.output).stem + "_benchmark.csv"))
            export_benchmark_csv(benchmark_rows, benchmark_output)
            print_benchmark_summary(benchmark_rows)
            print(f"Exported benchmark summary to {benchmark_output}")
        print(f"\nExported {len(parsed_results)} results to {args.output}")
