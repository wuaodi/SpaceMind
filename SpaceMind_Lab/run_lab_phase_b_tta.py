#!/usr/bin/env python3
from __future__ import annotations

import argparse

from lab_experiment_common import (
    BENCHMARK_DIR,
    DEFAULT_TTA_ROUNDS,
    build_auto_record,
    build_manual_record,
    build_phase_b_tta_summary,
    collect_final_metadata,
    collect_initial_metadata,
    ensure_benchmark_dir,
    export_benchmark_rows,
    join_manual_auto,
    load_csv_records,
    rank_navigation_combos,
    run_host,
    single_run_worse_key,
    wait_for_operator,
    write_records_csv_jsonl,
)
from runtime_logs.result_parser import export_csv


def select_worst_combo(phase_a_manual: list[dict], phase_a_auto: list[dict]) -> tuple[dict, dict]:
    ranked = rank_navigation_combos(phase_a_manual, phase_a_auto)
    if not ranked:
        raise RuntimeError("No Phase A navigation records found. Run run_lab_phase_a.py first.")

    worst_combo = ranked[0]
    joined = join_manual_auto(phase_a_manual, phase_a_auto)
    candidates = [
        item
        for item in joined
        if item.get("satellite") == worst_combo["satellite"]
        and item.get("task") == worst_combo["task"]
        and item.get("mode") == worst_combo["mode"]
    ]
    candidates.sort(key=single_run_worse_key)
    if not candidates:
        raise RuntimeError("Worst combo selected, but no matching manual rows were found.")
    return worst_combo, candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab Phase B TTA runner")
    parser.add_argument("--model", default="", help="Override model alias passed to host.py")
    parser.add_argument("--tta-rounds", type=int, default=DEFAULT_TTA_ROUNDS, help="Number of TTA rounds")
    args = parser.parse_args()

    ensure_benchmark_dir()

    phase_a_manual_csv = BENCHMARK_DIR / "lab_phase_a_manual.csv"
    phase_a_auto_csv = BENCHMARK_DIR / "lab_phase_a_auto.csv"
    phase_b_manual_csv = BENCHMARK_DIR / "lab_phase_b_tta_manual.csv"
    phase_b_manual_jsonl = BENCHMARK_DIR / "lab_phase_b_tta_manual.jsonl"
    phase_b_auto_csv = BENCHMARK_DIR / "lab_phase_b_tta_auto.csv"
    phase_b_auto_jsonl = BENCHMARK_DIR / "lab_phase_b_tta_auto.jsonl"
    phase_b_summary_csv = BENCHMARK_DIR / "lab_phase_b_tta_summary.csv"
    phase_b_benchmark_csv = BENCHMARK_DIR / "lab_phase_b_tta_auto_benchmark.csv"

    phase_a_manual = load_csv_records(phase_a_manual_csv)
    phase_a_auto = load_csv_records(phase_a_auto_csv)
    worst_combo, worst_condition = select_worst_combo(phase_a_manual, phase_a_auto)

    satellite = worst_combo["satellite"]
    task = worst_combo["task"]
    mode = worst_combo["mode"]
    condition_id = worst_condition["condition_id"]
    tta_workspace = f"tta_workspace/{satellite}/{task}/{mode}/hard_condition"

    print("Phase B selected worst navigation combo:")
    print(f"  satellite     : {satellite}")
    print(f"  task          : {task}")
    print(f"  mode          : {mode}")
    print(f"  condition     : {condition_id}")
    print(f"  success_rate  : {worst_combo['success_rate']}")
    print(f"  collision_rate: {worst_combo['collision_rate']}")
    print(f"  distance_error: {worst_combo['avg_distance_error_m']}")
    print(f"  avg_steps     : {worst_combo['avg_steps']}")
    print(f"  tta_workspace : {tta_workspace}")
    wait_for_operator("请确认以上组合就是今天要做 TTA 的对象。")

    manual_records: list[dict] = []
    auto_records: list[dict] = []

    for round_index in range(1, args.tta_rounds + 1):
        print(f"\n{'=' * 72}")
        print(f"[TTA {round_index}/{args.tta_rounds}] satellite={satellite} task={task} mode={mode} condition={condition_id}")
        print(f"{'=' * 72}")

        initial_meta = collect_initial_metadata(
            phase="Phase B TTA",
            satellite=satellite,
            task=task,
            mode=mode,
            condition_id=condition_id,
            tta_round=round_index,
        )
        wait_for_operator("请摆到这轮 TTA 的初始位姿，并确认系统已准备好。")

        host_returncode, log_path, auto_result = run_host(
            task=task,
            satellite=satellite,
            mode=mode,
            model=args.model,
            enable_tta=True,
            tta_workspace=tta_workspace,
        )
        log_file = log_path.name if log_path else ""
        print(f"host return code: {host_returncode}")
        print(f"log file: {log_file or '未检测到新日志'}")

        final_meta = collect_final_metadata(task=task)

        manual_record = build_manual_record(
            phase="phase_b_tta",
            run_index=round_index,
            satellite=satellite,
            task=task,
            mode=mode,
            condition_id=condition_id,
            log_file=log_file,
            host_returncode=host_returncode,
            initial_meta=initial_meta,
            final_meta=final_meta,
            enable_tta=True,
            tta_workspace=tta_workspace,
            tta_round=round_index,
        )
        auto_record = build_auto_record(
            phase="phase_b_tta",
            satellite=satellite,
            task=task,
            mode=mode,
            condition_id=condition_id,
            log_file=log_file,
            host_returncode=host_returncode,
            auto_record=auto_result,
            enable_tta=True,
            tta_workspace=tta_workspace,
        )
        auto_record["tta_round"] = round_index

        manual_records.append(manual_record)
        auto_records.append(auto_record)
        write_records_csv_jsonl(phase_b_manual_csv, phase_b_manual_jsonl, manual_records)
        write_records_csv_jsonl(phase_b_auto_csv, phase_b_auto_jsonl, auto_records)

        summary_rows = build_phase_b_tta_summary(manual_records, auto_records)
        if summary_rows:
            export_csv(summary_rows, str(phase_b_summary_csv))
        export_benchmark_rows(auto_records, phase_b_benchmark_csv)

    print(f"\nPhase B TTA complete: {len(manual_records)} runs")
    print(f"Manual CSV: {phase_b_manual_csv}")
    print(f"Auto CSV: {phase_b_auto_csv}")
    print(f"Summary CSV: {phase_b_summary_csv}")
    print(f"Auto benchmark CSV: {phase_b_benchmark_csv}")


if __name__ == "__main__":
    main()
