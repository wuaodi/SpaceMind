# Phase A 工具配置筛选 批量脚本
# 用法：在 SpaceMind_UE5 目录下运行
#   python run_phase_a.py
#
# 共 9 条实验：3 任务 × 3 profile
#   rendezvous-hold-front × {vision_only, hybrid_nav, hybrid_nav_with_code}
#   search-then-approach  × {vision_only, hybrid_nav, hybrid_nav_with_code}
#   inspection-diagnosis  × {vision_only, hybrid_nav, hybrid_nav_with_code}

import subprocess
import sys
import time

FLY_SCRIPT = "environments/satellite_pipeline/fly_redis.py"
WAIT_SECONDS = 60
MAX_RETRIES = 2
PROFILES = ["vision_only", "hybrid_nav", "hybrid_nav_with_code"]

TASKS = [
    {
        "name": "rendezvous-hold-front",
        "fly_args": ["--init_x", "-11", "--init_y", "0", "--init_z", "0"],
        "host_extra": [],
    },
    {
        "name": "search-then-approach",
        "fly_args": ["--init_x", "-11", "--init_y", "0", "--init_z", "0", "--init_yaw", "0.5"],
        "host_extra": [],
    },
    {
        "name": "inspection-diagnosis",
        "fly_args": ["--init_x", "-5", "--init_y", "0", "--init_z", "0"],
        "host_extra": ["--target_name", "CAPSTONE"],
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


def run_one(task_cfg, profile, run_idx, total):
    task_name = task_cfg["name"]
    fly_args = task_cfg["fly_args"]
    host_extra = task_cfg["host_extra"]

    for attempt in range(1, MAX_RETRIES + 1):
        tag = f"[{run_idx}/{total}]" if attempt == 1 else f"[{run_idx}/{total} retry {attempt}]"
        print(f"\n{'=' * 60}")
        print(f"{tag} task={task_name}  profile={profile}")
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
        ] + host_extra
        print(f"  host.py: {' '.join(host_cmd)}")
        result = subprocess.run(host_cmd)

        kill_fly(fly_proc)

        if result.returncode == 0:
            print(f"  {tag} OK: {task_name} + {profile}")
            return "ok"
        else:
            print(f"  {tag} FAILED (exit={result.returncode}): {task_name} + {profile}")
            if attempt < MAX_RETRIES:
                print(f"  Will retry in 5s...")
                time.sleep(5)

    return "fail"


def main():
    runs = [(t, p) for t in TASKS for p in PROFILES]
    total = len(runs)
    print(f"Phase A: {total} runs total")
    print(f"  Tasks: {[t['name'] for t in TASKS]}")
    print(f"  Profiles: {PROFILES}")

    results = []
    for i, (task_cfg, profile) in enumerate(runs, 1):
        status = run_one(task_cfg, profile, i, total)
        results.append((task_cfg["name"], profile, status))

    print(f"\n{'=' * 60}")
    print(f"Phase A complete. {total} runs.")
    ok = sum(1 for *_, s in results if s == "ok")
    fail = total - ok
    print(f"  OK: {ok}  FAIL: {fail}")
    if fail:
        for name, prof, st in results:
            if st != "ok":
                print(f"  FAILED: {name} + {prof}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
