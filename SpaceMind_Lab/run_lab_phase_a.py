#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from lab_experiment_common import (
    BENCHMARK_DIR,
    DEFAULT_MODE,
    DEFAULT_TOOL_PROFILE,
    MODE_FLAGS,
    SATELLITES,
    TASKS,
    build_auto_record,
    build_manual_record,
    build_phase_a_mode_summary,
    collect_final_metadata,
    collect_initial_metadata,
    ensure_benchmark_dir,
    export_benchmark_rows,
    rank_navigation_combos,
    run_host,
    wait_for_operator,
    write_records_csv_jsonl,
)
from runtime_logs.result_parser import export_csv

CONDITIONS = ["C1", "C2"]


def build_runs(smoke: bool) -> list[tuple[str, str, str, str]]:
    runs = []
    for satellite in SATELLITES:
        for task in TASKS:
            for mode in MODE_FLAGS:
                for condition_id in CONDITIONS:
                    runs.append((satellite, task, mode, condition_id))
    return runs[:1] if smoke else runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab Phase A batch runner")
    parser.add_argument("--model", default="", help="Override model alias passed to host.py")
    parser.add_argument("--smoke", action="store_true", help="Only run the first combination")
    args = parser.parse_args()

    ensure_benchmark_dir()

    manual_csv = BENCHMARK_DIR / "lab_phase_a_manual.csv"
    manual_jsonl = BENCHMARK_DIR / "lab_phase_a_manual.jsonl"
    auto_csv = BENCHMARK_DIR / "lab_phase_a_auto.csv"
    auto_jsonl = BENCHMARK_DIR / "lab_phase_a_auto.jsonl"
    mode_summary_csv = BENCHMARK_DIR / "lab_phase_a_mode_summary.csv"
    combo_summary_csv = BENCHMARK_DIR / "lab_phase_a_navigation_combo_summary.csv"
    benchmark_csv = BENCHMARK_DIR / "lab_phase_a_auto_benchmark.csv"

    runs = build_runs(args.smoke)
    total = len(runs)
    print(f"Phase A total runs: {total}")
    print(f"Satellites: {SATELLITES}")
    print(f"Tasks: {TASKS}")
    print(f"Modes: {list(MODE_FLAGS.keys())}")
    print(f"Conditions: {CONDITIONS}")
    print(f"Tool profile: {DEFAULT_TOOL_PROFILE}")
    print(f"Model override: {args.model or '(use default from .env/config)'}")

    manual_records: list[dict] = []
    auto_records: list[dict] = []

    for index, (satellite, task, mode, condition_id) in enumerate(runs, 1):
        print(f"\n{'=' * 72}")
        print(f"[{index}/{total}] satellite={satellite} task={task} mode={mode} condition={condition_id}")
        print(f"{'=' * 72}")

        initial_meta = collect_initial_metadata(
            phase="Phase A",
            satellite=satellite,
            task=task,
            mode=mode,
            condition_id=condition_id,
        )
        wait_for_operator("请按你刚才填写的初始条件摆位，并确认相机/桥接/Redis/执行器都已就绪。")

        host_returncode, log_path, auto_result = run_host(
            task=task,
            satellite=satellite,
            mode=mode,
            model=args.model,
        )
        log_file = log_path.name if log_path else ""
        print(f"host return code: {host_returncode}")
        print(f"log file: {log_file or '未检测到新日志'}")

        final_meta = collect_final_metadata(task=task)

        manual_record = build_manual_record(
            phase="phase_a",
            run_index=index,
            satellite=satellite,
            task=task,
            mode=mode,
            condition_id=condition_id,
            log_file=log_file,
            host_returncode=host_returncode,
            initial_meta=initial_meta,
            final_meta=final_meta,
        )
        auto_record = build_auto_record(
            phase="phase_a",
            satellite=satellite,
            task=task,
            mode=mode,
            condition_id=condition_id,
            log_file=log_file,
            host_returncode=host_returncode,
            auto_record=auto_result,
        )

        manual_records.append(manual_record)
        auto_records.append(auto_record)
        write_records_csv_jsonl(manual_csv, manual_jsonl, manual_records)
        write_records_csv_jsonl(auto_csv, auto_jsonl, auto_records)

        mode_summary = build_phase_a_mode_summary(manual_records, auto_records)
        combo_summary = rank_navigation_combos(manual_records, auto_records)
        if mode_summary:
            export_csv(mode_summary, str(mode_summary_csv))
        if combo_summary:
            export_csv(combo_summary, str(combo_summary_csv))
        export_benchmark_rows(auto_records, benchmark_csv)

    print(f"\nPhase A complete: {len(manual_records)} runs")
    print(f"Manual CSV: {manual_csv}")
    print(f"Auto CSV: {auto_csv}")
    print(f"Mode summary CSV: {mode_summary_csv}")
    print(f"Navigation combo summary CSV: {combo_summary_csv}")
    print(f"Auto benchmark CSV: {benchmark_csv}")


if __name__ == "__main__":
    main()
