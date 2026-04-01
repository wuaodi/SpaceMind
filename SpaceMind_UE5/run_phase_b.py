# Phase B 推理模式主实验 批量脚本（5 卫星 × 3 任务 × 3 模式 × 3 初始条件 = 135 runs）
# 用法：在 SpaceMind_UE5 目录下运行
#   python run_phase_b.py                          # 全自动续跑（需先在 UE5 中启动 ue_daemon）
#   python run_phase_b.py --satellite CAPSTONE     # 只跑指定卫星
#   python run_phase_b.py --condition C2           # 只跑指定条件
#   python run_phase_b.py --satellite IBEX --condition C3  # 组合过滤
#   python run_phase_b.py --manual                 # 手动模式（不依赖 daemon，每颗卫星间暂停等手动切换）
#   python run_phase_b.py --reset-done             # 清空完成记录，全部重跑
#
# 断点续跑：脚本维护 phase_b_done.txt 追踪已完成的实验，重启后自动跳过。
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
DONE_FILE = Path("phase_b_done.txt")
WAIT_SECONDS = 60
CUSTOM_DEPTH_WAIT = 20
MAX_RETRIES = 2
PROFILE = "hybrid_nav"

SATELLITES = ["CAPSTONE", "IBEX", "BioSentinel", "New_Horizons", "Huygens"]

TASK_NAMES = ["rendezvous-hold-front", "search-then-approach", "inspection-diagnosis"]

MODES = [
    {"name": "standard", "flags": []},
    {"name": "react", "flags": ["--enable_react"]},
    {"name": "world_model", "flags": ["--enable_world_model"]},
]

INIT_CONDITIONS = {
    "C1": {
        "label": "normal",
        "rendezvous-hold-front":   {"x": -11, "y": 0, "z": 0, "yaw": 0},
        "search-then-approach":    {"x": -11, "y": 0, "z": 0, "yaw": 0.5},
        "inspection-diagnosis":    {"x": -5,  "y": 0, "z": 0, "yaw": 0, "exposure": 0},
    },
    "C2": {
        "label": "far_offset",
        "rendezvous-hold-front":   {"x": -15, "y": 2, "z": -1, "yaw": 0},
        "search-then-approach":    {"x": -15, "y": 2, "z": -1, "yaw": 0.8},
        "inspection-diagnosis":    {"x": -7,  "y": 2, "z": -1, "yaw": 0, "exposure": -2},
    },
    "C3": {
        "label": "offset_alt",
        "rendezvous-hold-front":   {"x": -11, "y": -2, "z": 1, "yaw": 0},
        "search-then-approach":    {"x": -11, "y": -2, "z": 1, "yaw": -0.5},
        "inspection-diagnosis":    {"x": -5,  "y": -1, "z": 1, "yaw": 0, "exposure": 2},
    },
}


# --------------- done tracking ---------------

def _done_key(cond, sat, task, mode):
    return f"{cond}|{sat}|{task}|{mode}"


def load_done():
    done = set()
    if DONE_FILE.exists():
        for line in DONE_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                done.add(line)
    return done


def mark_done(cond, sat, task, mode):
    with DONE_FILE.open("a", encoding="utf-8") as f:
        f.write(_done_key(cond, sat, task, mode) + "\n")


def is_done(done_set, cond, sat, task, mode):
    return _done_key(cond, sat, task, mode) in done_set


def init_done_file():
    """Pre-populate done file with all known completed C1 runs + C2 CAPSTONE front standard."""
    if DONE_FILE.exists():
        return
    lines = ["# Phase B completed runs (auto-generated)"]
    for sat in SATELLITES:
        for task in TASK_NAMES:
            for m in MODES:
                lines.append(_done_key("C1", sat, task, m["name"]))
    lines.append(_done_key("C2", "CAPSTONE", "rendezvous-hold-front", "standard"))
    DONE_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  [done] Initialized {DONE_FILE} with {len(lines) - 1} completed entries")


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
    """Send swap command to UE daemon and wait until PIE is up + custom depth set."""
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
    """Tell UE daemon to stop PIE."""
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


def build_fly_args(cond_key, task_name):
    cond = INIT_CONDITIONS[cond_key]
    pos = cond[task_name]
    args = [
        "--init_x", str(pos["x"]),
        "--init_y", str(pos["y"]),
        "--init_z", str(pos["z"]),
    ]
    if pos.get("yaw", 0) != 0:
        args += ["--init_yaw", str(pos["yaw"])]
    exposure = pos.get("exposure", 0)
    if exposure != 0:
        args += ["--init_exposure", str(exposure)]
    return args


def run_one(satellite, task_name, mode_cfg, cond_key, run_idx, total):
    mode_name = mode_cfg["name"]
    mode_flags = mode_cfg["flags"]
    fly_args = build_fly_args(cond_key, task_name)
    cond_label = INIT_CONDITIONS[cond_key]["label"]

    for attempt in range(1, MAX_RETRIES + 1):
        tag = f"[{run_idx}/{total}]" if attempt == 1 else f"[{run_idx}/{total} retry {attempt}]"
        print(f"\n{'=' * 60}")
        print(f"{tag} sat={satellite}  task={task_name}  mode={mode_name}  cond={cond_key}({cond_label})")
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
    parser = argparse.ArgumentParser(description="Phase B: Reasoning Mode Ablation (Expanded)")
    parser.add_argument("--satellite", default=None, help="Only run for this satellite")
    parser.add_argument("--condition", default=None, help="Only run for this condition (C1/C2/C3)")
    parser.add_argument("--manual", action="store_true",
                        help="Manual mode: pause between satellites for hand-switching (no daemon needed)")
    parser.add_argument("--reset-done", action="store_true",
                        help="Clear done tracking file and re-run everything")
    args = parser.parse_args()

    if args.reset_done:
        _rm(DONE_FILE)
        print(f"  [done] Cleared {DONE_FILE}")

    init_done_file()
    done_set = load_done()
    print(f"  [done] Loaded {len(done_set)} completed entries from {DONE_FILE}")

    satellites = [args.satellite] if args.satellite else SATELLITES
    conditions = [args.condition] if args.condition else list(INIT_CONDITIONS.keys())

    all_combos = []
    for sat in satellites:
        for cond_key in conditions:
            for task_name in TASK_NAMES:
                for mode_cfg in MODES:
                    all_combos.append((sat, cond_key, task_name, mode_cfg))

    pending = [(s, c, t, m) for s, c, t, m in all_combos
               if not is_done(done_set, c, s, t, m["name"])]
    total = len(all_combos)
    skipped = total - len(pending)

    mode_label = "MANUAL" if args.manual else "AUTO (ue_daemon)"
    print(f"\nPhase B: {total} total, {skipped} skipped (done), {len(pending)} to run  [{mode_label}]")
    print(f"  Satellites: {satellites}")
    print(f"  Conditions: {conditions}")
    print(f"  Profile: {PROFILE}")

    if not pending:
        print("  All runs already completed!")
        return

    if not args.manual:
        TMP_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    run_idx = 0
    current_sat = None

    for sat, cond_key, task_name, mode_cfg in pending:
        if sat != current_sat:
            if current_sat is not None and not args.manual:
                signal_done()

            current_sat = sat
            remaining_for_sat = sum(1 for s, c, t, m in pending if s == sat)
            print(f"\n{'#' * 60}")
            print(f"  SATELLITE: {sat}  ({remaining_for_sat} runs remaining)")
            print(f"{'#' * 60}")

            if args.manual:
                input(f"\n  >>> Press ENTER when '{sat}' is ready in UE5 ... ")
            else:
                swap_satellite(sat)

        run_idx += 1
        status = run_one(sat, task_name, mode_cfg, cond_key, run_idx, len(pending))

        if status == "ok":
            mark_done(cond_key, sat, task_name, mode_cfg["name"])
            done_set.add(_done_key(cond_key, sat, task_name, mode_cfg["name"]))

        all_results.append((sat, task_name, mode_cfg["name"], cond_key, status))

    if current_sat is not None and not args.manual:
        signal_done()

    print(f"\n{'=' * 60}")
    print(f"Phase B batch complete. {len(pending)} runs attempted ({skipped} skipped).")
    ok = sum(1 for *_, s in all_results if s == "ok")
    fail = len(all_results) - ok
    print(f"  OK: {ok}  FAIL: {fail}")
    if fail:
        for sat, task, mode, cond, st in all_results:
            if st != "ok":
                print(f"  FAILED: {sat} / {task} / {mode} / {cond}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
