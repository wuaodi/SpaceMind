# Phase C 多目标泛化正式实验 批量脚本
# 用法：在 SpaceMind_UE5 目录下运行
#   python run_phase_c.py                        # 交互模式，5 颗卫星各 9 runs
#   python run_phase_c.py --satellite IBEX       # 只跑指定卫星
#   python run_phase_c.py --profile hybrid_nav   # 覆盖 profile（默认 hybrid_nav）
#   python run_phase_c.py --mode react           # 覆盖 mode（默认 standard）
#
# 共 45 条实验：5 卫星 × 3 任务 × 3 重复
#   卫星: CAPSTONE, IBEX, BioSentinel, New_Horizons, Huygens
#   任务: rendezvous-hold-front, search-then-approach, inspection-diagnosis
# 固定 Phase A+B 选出的最优 tool_profile 和 mode
#
# 注意：切换卫星需要在 UE5 Editor Python 控制台执行：
#   import os; os.environ["SPACEMIND_TARGET_SATELLITE"] = "IBEX"
#   import ue_place_satellite as sat_tool; import importlib; importlib.reload(sat_tool)
#   sat_tool.place_selected_satellite()

import argparse
import subprocess
import sys
import time

FLY_SCRIPT = "environments/satellite_pipeline/fly_redis.py"
WAIT_SECONDS = 60
MAX_RETRIES = 2
REPEATS = 3

SATELLITES = ["CAPSTONE", "IBEX", "BioSentinel", "New_Horizons", "Huygens"]

TASKS = [
    {
        "name": "rendezvous-hold-front",
        "fly_args": ["--init_x", "-11", "--init_y", "0", "--init_z", "0"],
    },
    {
        "name": "search-then-approach",
        "fly_args": ["--init_x", "-11", "--init_y", "0", "--init_z", "0", "--init_yaw", "0.5"],
    },
    {
        "name": "inspection-diagnosis",
        "fly_args": ["--init_x", "-5", "--init_y", "0", "--init_z", "0"],
    },
]


def kill_fly(fly_proc):
    if fly_proc is None:
        return
    fly_proc.terminate()
    try:
        fly_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        fly_proc.kill()
    time.sleep(3)


def run_one(satellite, task_cfg, profile, mode_flags, repeat_idx, run_idx, total):
    task_name = task_cfg["name"]
    fly_args = task_cfg["fly_args"]

    for attempt in range(1, MAX_RETRIES + 1):
        tag = f"[{run_idx}/{total}]" if attempt == 1 else f"[{run_idx}/{total} retry {attempt}]"
        print(f"\n{'=' * 60}")
        print(f"{tag} sat={satellite}  task={task_name}  repeat={repeat_idx}")
        print(f"{'=' * 60}")

        fly_cmd = [sys.executable, FLY_SCRIPT] + fly_args
        print(f"  fly_redis: {' '.join(fly_args)}")
        fly_proc = subprocess.Popen(fly_cmd)

        print(f"  Waiting {WAIT_SECONDS}s for environment...")
        time.sleep(WAIT_SECONDS)

        host_cmd = [
            sys.executable, "host.py",
            "--task", task_name,
            "--tool_profile", profile,
            "--target_name", satellite,
        ] + mode_flags
        print(f"  host.py: {' '.join(host_cmd[1:])}")
        result = subprocess.run(host_cmd)

        kill_fly(fly_proc)

        if result.returncode == 0:
            print(f"  {tag} OK")
            return "ok"
        else:
            print(f"  {tag} FAILED (exit={result.returncode})")
            if attempt < MAX_RETRIES:
                print("  Will retry in 5s...")
                time.sleep(5)

    return "fail"


def run_satellite_group(satellite, profile, mode_flags, global_idx, total):
    results = []
    for task_cfg in TASKS:
        for r in range(1, REPEATS + 1):
            status = run_one(satellite, task_cfg, profile, mode_flags, r, global_idx, total)
            results.append((satellite, task_cfg["name"], r, status))
            global_idx += 1
    return results, global_idx


def main():
    parser = argparse.ArgumentParser(description="Phase C: Multi-Target Generalization")
    parser.add_argument("--satellite", default=None,
                        help="Only run experiments for this satellite (skip others)")
    parser.add_argument("--profile", default="hybrid_nav",
                        help="Tool profile to use (default: hybrid_nav from Phase A)")
    parser.add_argument("--mode", default="standard", choices=["standard", "react", "world_model"],
                        help="Reasoning mode (default: standard from Phase B)")
    args = parser.parse_args()

    mode_flags = []
    if args.mode == "react":
        mode_flags = ["--enable_react"]
    elif args.mode == "world_model":
        mode_flags = ["--enable_world_model"]

    satellites = [args.satellite] if args.satellite else SATELLITES
    runs_per_sat = len(TASKS) * REPEATS
    total = runs_per_sat * len(satellites)

    print(f"Phase C: {total} runs total ({runs_per_sat} per satellite)")
    print(f"  Satellites: {satellites}")
    print(f"  Tasks: {[t['name'] for t in TASKS]}")
    print(f"  Repeats: {REPEATS}")
    print(f"  Profile: {args.profile}  Mode: {args.mode}")

    all_results = []
    global_idx = 1

    for i, sat in enumerate(satellites):
        print(f"\n{'#' * 60}")
        print(f"  SATELLITE GROUP {i+1}/{len(satellites)}: {sat}")
        print(f"  Ensure '{sat}' is placed in UE5 before continuing.")
        print(f"{'#' * 60}")

        if i > 0 or (args.satellite is None and len(satellites) > 1 and i == 0):
            input(f"\n  >>> Press ENTER when '{sat}' is ready in UE5 ... ")

        results, global_idx = run_satellite_group(sat, args.profile, mode_flags, global_idx, total)
        all_results.extend(results)

    print(f"\n{'=' * 60}")
    print(f"Phase C complete. {total} runs.")
    ok = sum(1 for *_, s in all_results if s == "ok")
    fail = total - ok
    print(f"  OK: {ok}  FAIL: {fail}")
    if fail:
        for sat, task, rep, st in all_results:
            if st != "ok":
                print(f"  FAILED: {sat} / {task} / repeat {rep}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
