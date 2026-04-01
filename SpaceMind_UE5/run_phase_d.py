# Phase D TTA 在线学习 批量脚本（6 组 × 5 轮 = 30 runs）
# 用法：在 SpaceMind_UE5 目录下运行
#   python run_phase_d.py                    # 全自动续跑（需先在 UE5 中启动 ue_daemon）
#   python run_phase_d.py --manual           # 手动模式（每颗卫星间暂停手动切换）
#   python run_phase_d.py --group G1         # 只跑某一组
#   python run_phase_d.py --group G1 --group G2  # 跑多组
#
# 6 组任务（基于 Phase B 结果精选，覆盖 front/search/inspection 三种任务类型）：
#   CAPSTONE:     G1 search C1 (基线 OK 2.597m/14步)  + G2 inspection C2 (基线 50分)
#   New_Horizons: G3 search C1 (基线 FAIL 10步)       + G4 front C3 (基线 FAIL 20步)
#   Huygens:      G5 front C2 (基线 OK 2.209m/8步)    + G6 inspection C1 (基线 12分)
#
# 每组使用独立 tta_workspace，连续 5 轮共享经验。
#
# 自动模式前置条件：
#   在 UE5 Editor -> Output Log -> Python 中执行：
#   import sys
#   sys.path.append(r"D:\project\SpaceMind\SpaceMind_UE5\environments\satellite_pipeline")
#   import ue_daemon as d
#   d.start()

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

FLY_SCRIPT = "environments/satellite_pipeline/fly_redis.py"
CUSTOM_DEPTH_SCRIPT = "environments/satellite_pipeline/set_custom_depth_airsim.py"
TMP_DIR = Path("environments/satellite_pipeline/_tmp")
WAIT_SECONDS = 60
CUSTOM_DEPTH_WAIT = 20
MAX_RETRIES = 2
TTA_ROUNDS = 5
PROFILE = "hybrid_nav"

TASK_GROUPS = [
    {
        "id": "G1", "satellite": "CAPSTONE",
        "task": "search-then-approach", "condition": "C1", "mode": "standard",
        "fly_args": ["--init_x", "-11", "--init_y", "0", "--init_z", "0", "--init_yaw", "0.5"],
        "tta_workspace": "tta_d2_capstone_search_c1",
        "baseline": "OK (2.597m, 14步)",
    },
    {
        "id": "G2", "satellite": "CAPSTONE",
        "task": "inspection-diagnosis", "condition": "C2", "mode": "standard",
        "fly_args": ["--init_x", "-7", "--init_y", "2", "--init_z", "-1", "--init_exposure", "-2"],
        "tta_workspace": "tta_d2_capstone_insp_c2",
        "baseline": "50分",
    },
    {
        "id": "G3", "satellite": "New_Horizons",
        "task": "search-then-approach", "condition": "C1", "mode": "standard",
        "fly_args": ["--init_x", "-11", "--init_y", "0", "--init_z", "0", "--init_yaw", "0.5"],
        "tta_workspace": "tta_d2_newhorizons_search_c1",
        "baseline": "FAIL (10步, missing_lidar)",
    },
    {
        "id": "G4", "satellite": "New_Horizons",
        "task": "rendezvous-hold-front", "condition": "C3", "mode": "standard",
        "fly_args": ["--init_x", "-11", "--init_y", "-2", "--init_z", "1"],
        "tta_workspace": "tta_d2_newhorizons_front_c3",
        "baseline": "FAIL (20步, missing_lidar)",
    },
    {
        "id": "G5", "satellite": "Huygens",
        "task": "rendezvous-hold-front", "condition": "C2", "mode": "standard",
        "fly_args": ["--init_x", "-15", "--init_y", "2", "--init_z", "-1"],
        "tta_workspace": "tta_d2_huygens_front_c2",
        "baseline": "OK (2.209m, 8步)",
    },
    {
        "id": "G6", "satellite": "Huygens",
        "task": "inspection-diagnosis", "condition": "C1", "mode": "standard",
        "fly_args": ["--init_x", "-5", "--init_y", "0", "--init_z", "0"],
        "tta_workspace": "tta_d2_huygens_insp_c1",
        "baseline": "12分",
    },
]


# --------------- file IPC helpers ---------------

def wait_for_file(path, timeout_s=180):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if path.exists():
            return
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {path} ({timeout_s}s)")


def _rm(p):
    try:
        p.unlink()
    except Exception:
        pass


def swap_satellite(sat_name):
    idle_flag = TMP_DIR / "idle.flag"
    ready_flag = TMP_DIR / "ready.flag"
    cmd_path = TMP_DIR / "ue_cmd.json"
    session_dir = TMP_DIR / f"session_{sat_name}"
    session_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = session_dir / "name_label_mapping.txt"

    print(f"  [auto] Waiting for UE daemon idle...")
    wait_for_file(idle_flag, timeout_s=600)
    _rm(idle_flag)
    _rm(ready_flag)
    _rm(TMP_DIR / "done.flag")

    cmd = {
        "sat_name": sat_name,
        "mapping_out": str(mapping_path.resolve()).replace("\\", "/"),
    }
    cmd_path.write_text(json.dumps(cmd, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [auto] Sent swap command for '{sat_name}', waiting for PIE ready...")

    wait_for_file(ready_flag, timeout_s=180)
    print(f"  [auto] PIE ready. Setting custom depth...")

    subprocess.run(
        [sys.executable, CUSTOM_DEPTH_SCRIPT, "--mapping_file", str(mapping_path.resolve())],
        check=False,
    )
    print(f"  [auto] Waiting {CUSTOM_DEPTH_WAIT}s for custom depth to take effect...")
    time.sleep(CUSTOM_DEPTH_WAIT)
    print(f"  [auto] Satellite '{sat_name}' ready.")


def signal_done():
    done_flag = TMP_DIR / "done.flag"
    done_flag.write_text("ok\n", encoding="utf-8")
    print(f"  [auto] Sent done signal, waiting for UE daemon to stop PIE...")
    wait_for_file(TMP_DIR / "idle.flag", timeout_s=60)
    print(f"  [auto] UE daemon idle.")


# --------------- experiment helpers ---------------

def kill_fly(fly_proc):
    if fly_proc is None:
        return
    fly_proc.terminate()
    try:
        fly_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        fly_proc.kill()
    time.sleep(3)


def run_one(group, tta_round, run_idx, total):
    task_name = group["task"]
    fly_args = group["fly_args"]
    tta_ws = group["tta_workspace"]
    satellite = group["satellite"]

    mode_flags = []
    if group["mode"] == "react":
        mode_flags = ["--enable_react"]
    elif group["mode"] == "world_model":
        mode_flags = ["--enable_world_model"]

    for attempt in range(1, MAX_RETRIES + 1):
        tag = f"[{run_idx}/{total}]" if attempt == 1 else f"[{run_idx}/{total} retry {attempt}]"
        print(f"\n{'=' * 60}")
        print(f"{tag} {group['id']}  sat={satellite}  task={task_name}  round={tta_round}/{TTA_ROUNDS}  ws={tta_ws}")
        print(f"{'=' * 60}")

        fly_cmd = [sys.executable, FLY_SCRIPT] + fly_args
        print(f"  fly_redis: {' '.join(fly_args)}")
        fly_proc = subprocess.Popen(fly_cmd)

        print(f"  Waiting {WAIT_SECONDS}s for environment...")
        time.sleep(WAIT_SECONDS)

        host_cmd = [
            sys.executable, "host.py",
            "--task", task_name,
            "--tool_profile", PROFILE,
            "--target_name", satellite,
            "--enable_tta",
            "--tta_workspace", tta_ws,
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


# --------------- main ---------------

def main():
    parser = argparse.ArgumentParser(description="Phase D: TTA Online Learning (6 groups x 5 rounds)")
    parser.add_argument("--group", action="append", default=None,
                        help="Only run specified group(s), e.g. --group G1 --group G3")
    parser.add_argument("--manual", action="store_true",
                        help="Manual mode: pause between satellites for hand-switching")
    args = parser.parse_args()

    if args.group:
        selected_ids = {g.upper() for g in args.group}
        groups = [g for g in TASK_GROUPS if g["id"] in selected_ids]
        if not groups:
            print(f"  No matching groups for {args.group}. Available: {[g['id'] for g in TASK_GROUPS]}")
            return
    else:
        groups = TASK_GROUPS

    mode_label = "MANUAL" if args.manual else "AUTO (ue_daemon)"
    total = len(groups) * TTA_ROUNDS
    print(f"Phase D TTA: {total} runs total  [{mode_label}]")
    for g in groups:
        print(f"  {g['id']}: {g['satellite']} / {g['task']} / {g['condition']} / {g['mode']}  baseline={g['baseline']}")
    print(f"  Profile: {PROFILE}  TTA rounds: {TTA_ROUNDS}")

    if not args.manual:
        TMP_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    global_idx = 0
    current_sat = None

    for group in groups:
        sat = group["satellite"]

        if sat != current_sat:
            if current_sat is not None and not args.manual:
                signal_done()

            current_sat = sat
            print(f"\n{'#' * 60}")
            print(f"  SATELLITE: {sat}")
            print(f"{'#' * 60}")

            if args.manual:
                input(f"\n  >>> Press ENTER when '{sat}' is ready in UE5 ... ")
            else:
                swap_satellite(sat)

        task_name = group["task"]
        tta_ws = group["tta_workspace"]
        print(f"\n{'~' * 60}")
        print(f"  TTA GROUP {group['id']}: {task_name} ({group['condition']})  workspace: {tta_ws}")
        print(f"  Phase B baseline: {group['baseline']}")
        print(f"  Running {TTA_ROUNDS} consecutive rounds with shared experience")
        print(f"{'~' * 60}")

        for tta_round in range(1, TTA_ROUNDS + 1):
            global_idx += 1
            status = run_one(group, tta_round, global_idx, total)
            all_results.append((group["id"], task_name, tta_round, status))

    if current_sat is not None and not args.manual:
        signal_done()

    print(f"\n{'=' * 60}")
    print(f"Phase D TTA complete. {total} runs.")
    ok = sum(1 for *_, s in all_results if s == "ok")
    fail = total - ok
    print(f"  OK: {ok}  FAIL: {fail}")

    for group in groups:
        gid = group["id"]
        group_results = [(r, s) for g, _, r, s in all_results if g == gid]
        curve = " -> ".join(s for _, s in group_results)
        print(f"  {gid} ({group['satellite']} {group['task']} {group['condition']}): {curve}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
