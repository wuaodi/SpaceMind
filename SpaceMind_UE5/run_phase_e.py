# Phase E 挑战场景批量脚本（E1 旋转目标 / E2 Δv 预算 / E3 执行退化 / E4 感知退化）
# 用法：在 SpaceMind_UE5 目录下运行（需先在 UE5 中启动 ue_daemon，参照 run_phase_b.py 顶部说明）
#   python run_phase_e.py                 # 全自动：按 P0 -> P1 -> P2 优先级消化全部队列，断点续跑
#   python run_phase_e.py --smoke         # 冒烟测试：6 个 run（每类场景各 1），验证全链路，不写 done 记录
#   python run_phase_e.py --list          # 只打印队列不运行
#   python run_phase_e.py --priority P0   # 只跑 P0
#   python run_phase_e.py --scenario E1 --satellite CAPSTONE
#   python run_phase_e.py --reset-done    # 清空完成记录
#
# 场景设计：
#   E1 旋转目标（P0/P1）：target_motion.py 以固定角速率旋转目标星（R1=0.5°/s 温和, R2=2.0°/s 困难），
#      任务 front + inspection，初始条件用 C1 标称位，把挑战变量隔离到目标运动上。
#   E2 Δv 预算（P0/P1）：host.py --deltav_budget，预算 = 1.5 × (初始直线距离 - 2m)，
#      初始条件用 C2 远距偏移位，任务 front + search。
#   E3N 执行噪声 / E3F 推力故障 / E4L LiDAR 间歇失效 / E4E 中途曝光扰动（P2）：
#      fly_redis.py 注入开关，任务 front，初始条件用 C1 标称位，把挑战变量隔离到注入本身。
#
# 每个 run 结束后自动用 result_parser 解析日志，追加到 phase_e_results.jsonl（自动评分闭环）。

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# 子进程（set_custom_depth 等）打印 ✓/✗ 在 GBK 控制台会崩，强制全链路 UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

from runtime_logs.result_parser import parse_single_log

FLY_SCRIPT = "environments/satellite_pipeline/fly_redis.py"
MOTION_SCRIPT = "environments/satellite_pipeline/target_motion.py"
CUSTOM_DEPTH_SCRIPT = "environments/satellite_pipeline/set_custom_depth_airsim.py"
TMP_DIR = Path("environments/satellite_pipeline/_tmp")
LOG_DIR = Path("runtime_logs/log")
DONE_FILE = Path("phase_e_done.txt")
RESULTS_FILE = Path("phase_e_results.jsonl")
WAIT_SECONDS = 60
CUSTOM_DEPTH_WAIT = 20
MAX_RETRIES = 2
PROFILE = "hybrid_nav"

SATELLITES = ["CAPSTONE", "IBEX", "BioSentinel", "New_Horizons", "Huygens"]

MODE_FLAGS = {
    "standard": [],
    "world_model": ["--enable_world_model"],
    "react": ["--enable_react"],
}

E1_RATES = {"R1": 0.5, "R2": 2.0}  # 度/秒

# 初始条件（沿用 run_phase_b.py 的定义）
E1_INIT = {  # C1 标称位，挑战变量隔离到目标旋转
    "rendezvous-hold-front": {"x": -11, "y": 0, "z": 0},
    "inspection-diagnosis": {"x": -5, "y": 0, "z": 0},
}
E2_INIT = {  # C2 远距偏移位，预算更紧张
    "rendezvous-hold-front": {"x": -15, "y": 2, "z": -1},
    "search-then-approach": {"x": -15, "y": 2, "z": -1, "yaw": 0.8},
}
BUDGET_RATIO = 1.5

# E3/E4 退化注入场景：fly_redis 额外参数 + 写进结果记录的参数 + run_key 标签
DEGRADE_INIT = {"x": -11, "y": 0, "z": 0}  # C1 标称位
DEGRADE_SCENARIOS = {
    "E3N": {"fly": ["--noise"], "label": "noise0.1",
            "record": {"noise_pos_m": 0.1, "noise_att_rad": 0.02}},
    # 故障加在接近主轴 dx 上：front 任务沿 x 直线接近，dy/dz 指令常年为 0 触发不到
    "E3F": {"fly": ["--fault_axis", "dx", "--fault_scale", "0.5"], "label": "faultdx0.5",
            "record": {"fault_axis": "dx", "fault_scale": 0.5}},
    "E4L": {"fly": ["--lidar_dropout", "0.3"], "label": "dropout0.3",
            "record": {"lidar_dropout": 0.3}},
    "E4E": {"fly": ["--exposure_disturb_step", "3", "--exposure_disturb_value", "-3"], "label": "expstep3v-3",
            "record": {"exposure_disturb_step": 3, "exposure_disturb_value": -3.0}},
    # stage-2 加难变体（--stage2 时排队）
    "E3F2": {"fly": ["--fault_axis", "dx", "--fault_scale", "0.3"], "label": "faultdx0.3",
             "record": {"fault_axis": "dx", "fault_scale": 0.3}},
    "E4L2": {"fly": ["--lidar_dropout", "0.5"], "label": "dropout0.5",
             "record": {"lidar_dropout": 0.5}},
    "E4E2": {"fly": ["--exposure_disturb_step", "3", "--exposure_disturb_value", "3"], "label": "expstep3v+3",
             "record": {"exposure_disturb_step": 3, "exposure_disturb_value": 3.0}},
}


# 绕飞任务：C1 标称位出发，机动到目标正上方约 5m（NED 上方 -z），终局判定在 host 评测分支
FA_TASK = "fly-around-above"
FA_INIT = {"x": -11, "y": 0, "z": 0}
FA_GOAL_UP_M = 5.0


def e2_budget(task_name: str) -> float:
    if task_name == FA_TASK:
        # 预算 = 1.5 x 初始位到目标上方 5m 目标点 (0,0,-5) 的直线距离
        pos = FA_INIT
        dist = math.sqrt(pos["x"] ** 2 + pos["y"] ** 2 + (pos["z"] + FA_GOAL_UP_M) ** 2)
        return round(BUDGET_RATIO * dist, 1)
    pos = E2_INIT[task_name]
    dist = math.sqrt(pos["x"] ** 2 + pos["y"] ** 2 + pos["z"] ** 2)
    return round(BUDGET_RATIO * (dist - 2.0), 1)


BASE_DEGRADES = ["E3N", "E3F", "E4L", "E4E"]
STAGE2_SATELLITES = ["CAPSTONE", "Huygens"]  # stage-2 控制时长，只跑几何差异最大的两颗


def build_queue() -> list[dict]:
    """按 (priority, satellite) 分组生成全部 run。

    run dict 可叠加字段（组合场景直接组合字段即可）：
      rate_key  -> 启动 target_motion 旋转目标（可选 axis，默认 yaw）
      budget    -> host 加 --deltav_budget（按 E2_INIT 距离计算）
      degrade   -> fly_redis 注入参数场景名列表（DEGRADE_SCENARIOS）
    """
    runs = []
    for sat in SATELLITES:
        # P0: E1 front x 2 转速 x {standard, world_model}
        for rate_key in E1_RATES:
            for mode in ("standard", "world_model"):
                runs.append({"priority": "P0", "scenario": "E1", "satellite": sat,
                             "task": "rendezvous-hold-front", "mode": mode, "rate_key": rate_key})
        # P0: E2 front + search x {standard, world_model}
        for task in E2_INIT:
            for mode in ("standard", "world_model"):
                runs.append({"priority": "P0", "scenario": "E2", "satellite": sat,
                             "task": task, "mode": mode, "budget": True})
        # P1: E1 inspection x 2 转速 x {standard, world_model}
        for rate_key in E1_RATES:
            for mode in ("standard", "world_model"):
                runs.append({"priority": "P1", "scenario": "E1", "satellite": sat,
                             "task": "inspection-diagnosis", "mode": mode, "rate_key": rate_key})
        # P1: react 补全
        for task in E1_INIT:
            for rate_key in E1_RATES:
                runs.append({"priority": "P1", "scenario": "E1", "satellite": sat,
                             "task": task, "mode": "react", "rate_key": rate_key})
        for task in E2_INIT:
            runs.append({"priority": "P1", "scenario": "E2", "satellite": sat,
                         "task": task, "mode": "react", "budget": True})
        # P2: E3/E4 退化注入 x standard，任务 front
        for scen in BASE_DEGRADES:
            runs.append({"priority": "P2", "scenario": scen, "satellite": sat,
                         "task": "rendezvous-hold-front", "mode": "standard", "degrade": [scen]})

    sat_order = {s: i for i, s in enumerate(SATELLITES)}
    runs.sort(key=lambda r: (r["priority"], sat_order[r["satellite"]],
                             r["scenario"], r["task"], r.get("rate_key", ""), r["mode"]))
    return runs


def build_stage2_queue() -> list[dict]:
    """stage-2：加难 + 组合退化，P0-P2 指标未见退化时启用（--stage2）。"""
    runs = []
    front = "rendezvous-hold-front"
    for sat in STAGE2_SATELLITES:
        # E1 加难：R2 翻滚轴 x {standard, world_model}
        for mode in ("standard", "world_model"):
            runs.append({"priority": "P3", "scenario": "E1T", "satellite": sat,
                         "task": front, "mode": mode, "rate_key": "R2", "axis": "tumble"})
        # 单维加难
        for scen in ("E3F2", "E4L2", "E4E2"):
            runs.append({"priority": "P3", "scenario": scen, "satellite": sat,
                         "task": front, "mode": "standard", "degrade": [scen]})
        # 组合退化
        runs.append({"priority": "P3", "scenario": "E1xE2", "satellite": sat,
                     "task": front, "mode": "standard", "rate_key": "R1", "budget": True})
        runs.append({"priority": "P3", "scenario": "E3NxE4L", "satellite": sat,
                     "task": front, "mode": "standard", "degrade": ["E3N", "E4L"]})
        runs.append({"priority": "P3", "scenario": "E1xE4E", "satellite": sat,
                     "task": front, "mode": "standard", "rate_key": "R1", "degrade": ["E4E"]})
    return runs


def build_flyaround_queue() -> list[dict]:
    """绕飞任务队列（--flyaround）：正常条件 3 模式 + standard 下 6 种单维退化。"""
    runs = []
    for sat in SATELLITES:
        for mode in ("standard", "world_model", "react"):
            runs.append({"priority": "P4", "scenario": "FA", "satellite": sat,
                         "task": FA_TASK, "mode": mode})
        runs.append({"priority": "P4", "scenario": "FAxE1", "satellite": sat,
                     "task": FA_TASK, "mode": "standard", "rate_key": "R1"})
        runs.append({"priority": "P4", "scenario": "FAxE2", "satellite": sat,
                     "task": FA_TASK, "mode": "standard", "budget": True})
        for scen in BASE_DEGRADES:
            runs.append({"priority": "P4", "scenario": f"FAx{scen}", "satellite": sat,
                         "task": FA_TASK, "mode": "standard", "degrade": [scen]})
    return runs


def run_extras(run: dict) -> list[str]:
    extras = []
    if run.get("rate_key"):
        axis = run.get("axis", "yaw")
        extras.append(f"rate{E1_RATES[run['rate_key']]}" + ("" if axis == "yaw" else axis))
    if run.get("budget"):
        extras.append(f"budget{e2_budget(run['task'])}")
    for scen in run.get("degrade", []):
        extras.append(DEGRADE_SCENARIOS[scen]["label"])
    return extras


def run_key(run: dict) -> str:
    extra = "+".join(run_extras(run)) or "base"
    return f"{run['scenario']}|{run['satellite']}|{run['task']}|{run['mode']}|{extra}"


# --------------- done tracking ---------------

def load_done() -> set:
    if not DONE_FILE.exists():
        return set()
    return {line.strip() for line in DONE_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")}


def mark_done(key: str):
    with DONE_FILE.open("a", encoding="utf-8") as f:
        f.write(key + "\n")


# --------------- file IPC helpers（与 run_phase_b/d 一致）---------------

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

    print("  [auto] Waiting for UE daemon idle...")
    wait_for_file(idle_flag, timeout_s=600)
    _rm(idle_flag)
    _rm(ready_flag)
    _rm(TMP_DIR / "done.flag")

    cmd = {"sat_name": sat_name, "mapping_out": str(mapping_path.resolve()).replace("\\", "/")}
    cmd_path.write_text(json.dumps(cmd, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [auto] Sent swap command for '{sat_name}', waiting for PIE ready...")

    wait_for_file(ready_flag, timeout_s=180)
    print("  [auto] PIE ready. Setting custom depth...")
    subprocess.run([sys.executable, CUSTOM_DEPTH_SCRIPT, "--mapping_file", str(mapping_path.resolve())], check=False)
    print(f"  [auto] Waiting {CUSTOM_DEPTH_WAIT}s for custom depth to take effect...")
    time.sleep(CUSTOM_DEPTH_WAIT)
    print(f"  [auto] Satellite '{sat_name}' ready.")


def signal_done():
    done_flag = TMP_DIR / "done.flag"
    done_flag.write_text("ok\n", encoding="utf-8")
    print("  [auto] Sent done signal, waiting for UE daemon to stop PIE...")
    wait_for_file(TMP_DIR / "idle.flag", timeout_s=60)
    print("  [auto] UE daemon idle.")


# --------------- run helpers ---------------

def kill_proc(proc):
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(2)


def reset_target_orientation(sat: str) -> bool:
    result = subprocess.run(
        [sys.executable, MOTION_SCRIPT, "--satellite", sat, "--reset_only"],
        timeout=60,
    )
    return result.returncode == 0


def build_fly_args(init: dict) -> list:
    args = ["--init_x", str(init["x"]), "--init_y", str(init["y"]), "--init_z", str(init["z"])]
    if init.get("yaw", 0) != 0:
        args += ["--init_yaw", str(init["yaw"])]
    if init.get("exposure", 0) != 0:
        args += ["--init_exposure", str(init["exposure"])]
    return args


SMOKE_MODE = False  # main() 里按 --smoke 置位，冒烟记录打标便于统计时剔除


def record_result(run: dict, attempt: int, returncode: int, logs_before: set, started_at: str):
    """解析本次 run 新产生的日志，追加到 phase_e_results.jsonl。"""
    new_logs = sorted(set(LOG_DIR.glob("spacemind_*.txt")) - logs_before)
    record = {
        "phase": "E",
        "run_key": run_key(run),
        **({"smoke": True} if SMOKE_MODE else {}),
        "scenario": run["scenario"],
        "satellite": run["satellite"],
        "task": run["task"],
        "mode": run["mode"],
        "attempt": attempt,
        "host_returncode": returncode,
        "started_at": started_at,
        "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if run.get("rate_key"):
        record["rotation_rate_dps"] = E1_RATES[run["rate_key"]]
        record["rotation_axis"] = run.get("axis", "yaw")
    if run.get("budget"):
        record["deltav_budget_m"] = e2_budget(run["task"])
    for scen in run.get("degrade", []):
        record.update(DEGRADE_SCENARIOS[scen]["record"])
    if new_logs:
        parsed = parse_single_log(str(new_logs[-1]))
        record.update(parsed)
    else:
        record["log_file"] = ""
    with RESULTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_one(run: dict, run_idx: int, total: int) -> str:
    sat = run["satellite"]
    task_name = run["task"]
    # 初始位：绕飞任务固定 C1；预算场景用 C2 远距（预算公式基于它），旋转场景用 C1 标称，纯退化注入用 C1
    if task_name == FA_TASK:
        init = FA_INIT
    elif run.get("budget"):
        init = E2_INIT[task_name]
    elif run.get("rate_key"):
        init = E1_INIT[task_name]
    else:
        init = DEGRADE_INIT
    extra_desc = "+".join(run_extras(run)) or "base"

    for attempt in range(1, MAX_RETRIES + 1):
        tag = f"[{run_idx}/{total}]" if attempt == 1 else f"[{run_idx}/{total} retry {attempt}]"
        print(f"\n{'=' * 60}")
        print(f"{tag} {run['priority']} {run['scenario']}  sat={sat}  task={task_name}  mode={run['mode']}  {extra_desc}")
        print(f"{'=' * 60}")

        if not reset_target_orientation(sat):
            print(f"  {tag} FAILED: target orientation reset failed (mobility/mapping 问题?)")
            return "fail"

        fly_args = build_fly_args(init)
        for scen in run.get("degrade", []):
            fly_args += DEGRADE_SCENARIOS[scen]["fly"]
        print(f"  fly_redis: {' '.join(fly_args)}")
        fly_proc = subprocess.Popen([sys.executable, FLY_SCRIPT] + fly_args)

        motion_proc = None
        if run.get("rate_key"):
            rate = E1_RATES[run["rate_key"]]
            axis = run.get("axis", "yaw")
            motion_proc = subprocess.Popen(
                [sys.executable, MOTION_SCRIPT, "--satellite", sat, "--rate_dps", str(rate), "--axis", axis]
            )
            print(f"  target_motion: rate={rate} deg/s axis={axis}")

        print(f"  Waiting {WAIT_SECONDS}s for environment...")
        time.sleep(WAIT_SECONDS)

        if motion_proc is not None and motion_proc.poll() is not None:
            print(f"  {tag} FAILED: target_motion exited early (exit={motion_proc.returncode})")
            kill_proc(fly_proc)
            return "fail"

        host_cmd = [
            sys.executable, "host.py",
            "--task", task_name,
            "--tool_profile", PROFILE,
            "--target_name", sat,
        ] + MODE_FLAGS[run["mode"]]
        if run.get("budget"):
            host_cmd += ["--deltav_budget", str(e2_budget(task_name))]
        print(f"  host.py: {' '.join(host_cmd[1:])}")

        started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        logs_before = set(LOG_DIR.glob("spacemind_*.txt"))
        result = subprocess.run(host_cmd)

        kill_proc(motion_proc)
        kill_proc(fly_proc)
        record_result(run, attempt, result.returncode, logs_before, started_at)

        if result.returncode == 0:
            print(f"  {tag} OK")
            return "ok"
        print(f"  {tag} FAILED (exit={result.returncode})")
        if attempt < MAX_RETRIES:
            print("  Will retry in 5s...")
            time.sleep(5)

    return "fail"


# --------------- main ---------------

def main():
    parser = argparse.ArgumentParser(description="Phase E: challenging scenarios (rotating target + delta-v budget)")
    parser.add_argument("--smoke", action="store_true", help="冒烟测试：只跑 6 个 run")
    parser.add_argument("--list", action="store_true", help="只打印队列不运行")
    parser.add_argument("--priority", choices=["P0", "P1", "P2", "P3", "P4"], default=None)
    parser.add_argument("--scenario", default=None,
                        choices=["E1", "E2", "E1T", "E1xE2", "E3NxE4L", "E1xE4E",
                                 "FA", "FAxE1", "FAxE2", "FAxE3N", "FAxE3F", "FAxE4L", "FAxE4E"]
                                + list(DEGRADE_SCENARIOS))
    parser.add_argument("--satellite", choices=SATELLITES, default=None)
    parser.add_argument("--stage2", action="store_true", help="追加 stage-2 加难/组合退化队列")
    parser.add_argument("--flyaround", action="store_true", help="追加绕飞任务 P4 队列（15 正常 + 30 退化）")
    parser.add_argument("--manual", action="store_true", help="手动模式：每颗卫星间暂停等手动切换")
    parser.add_argument("--reset-done", action="store_true", help="清空完成记录后退出")
    args = parser.parse_args()

    if args.reset_done:
        _rm(DONE_FILE)
        print(f"Removed {DONE_FILE}")
        return

    global SMOKE_MODE
    SMOKE_MODE = args.smoke

    if args.smoke:
        if args.flyaround:
            runs = [
                {"priority": "P4", "scenario": "FA", "satellite": "CAPSTONE",
                 "task": FA_TASK, "mode": "standard"},
                {"priority": "P4", "scenario": "FAxE4L", "satellite": "CAPSTONE",
                 "task": FA_TASK, "mode": "standard", "degrade": ["E4L"]},
            ]
        else:
            runs = [
                {"priority": "P0", "scenario": "E1", "satellite": "CAPSTONE",
                 "task": "rendezvous-hold-front", "mode": "standard", "rate_key": "R1"},
                {"priority": "P0", "scenario": "E2", "satellite": "CAPSTONE",
                 "task": "rendezvous-hold-front", "mode": "standard", "budget": True},
            ] + [
                {"priority": "P2", "scenario": scen, "satellite": "CAPSTONE",
                 "task": "rendezvous-hold-front", "mode": "standard", "degrade": [scen]}
                for scen in BASE_DEGRADES
            ]
    else:
        runs = build_queue()
        if args.stage2:
            runs += build_stage2_queue()
        if args.flyaround:
            runs += build_flyaround_queue()
        done = load_done()
        if args.priority:
            runs = [r for r in runs if r["priority"] == args.priority]
        if args.scenario:
            runs = [r for r in runs if r["scenario"] == args.scenario]
        if args.satellite:
            runs = [r for r in runs if r["satellite"] == args.satellite]
        runs = [r for r in runs if run_key(r) not in done]

    total = len(runs)
    print(f"Phase E: {total} runs queued  [{'SMOKE' if args.smoke else 'MANUAL' if args.manual else 'AUTO (ue_daemon)'}]")
    for r in runs:
        print(f"  {r['priority']} {run_key(r)}")
    if args.list or total == 0:
        return

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    current_sat = None
    failed_swap_sats = set()  # 换星超时的卫星本轮跳过，done 文件保证重启后补跑

    try:
        for idx, run in enumerate(runs, 1):
            sat = run["satellite"]
            if sat in failed_swap_sats:
                results.append((run_key(run), "swap-skip"))
                continue
            if sat != current_sat:
                if current_sat is not None and not args.manual:
                    try:
                        signal_done()
                    except Exception as e:
                        print(f"  [auto] signal_done failed (continuing): {e}")
                print(f"\n{'#' * 60}\n  SATELLITE: {sat}\n{'#' * 60}")
                if args.manual:
                    input(f"\n  >>> Press ENTER when '{sat}' is ready in UE5 ... ")
                else:
                    try:
                        swap_satellite(sat)
                    except Exception as e:
                        print(f"  [auto] SWAP FAILED for '{sat}': {e} -> skip this satellite this session")
                        failed_swap_sats.add(sat)
                        current_sat = None
                        results.append((run_key(run), "swap-skip"))
                        continue
                current_sat = sat

            status = run_one(run, idx, total)
            results.append((run_key(run), status))
            if status == "ok" and not args.smoke:  # 冒烟不写 done，避免污染正式批量的断点续跑
                mark_done(run_key(run))
    finally:
        if current_sat is not None and not args.manual:
            try:
                signal_done()
            except Exception as e:
                print(f"  [auto] signal_done failed: {e}")

    ok = sum(1 for _, s in results if s == "ok")
    print(f"\n{'=' * 60}")
    print(f"Phase E finished this session: OK={ok} FAIL={len(results) - ok} (queued={total})")
    for key, status in results:
        print(f"  [{status.upper():4s}] {key}")
    print(f"Results appended to {RESULTS_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
