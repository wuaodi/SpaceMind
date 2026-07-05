# Phase E 目标星旋转控制器
"""
通过 AirSim simSetObjectPose 让目标卫星以固定角速率自旋/翻滚，模拟失稳目标。

原理：ue_place_satellite.py 把卫星各部件 StaticMeshActor 放在同一原点，
刚体旋转等于对每个部件施加相同的姿态（位置不变）。角度按 t 解析计算，
不做增量累积，部件之间不会漂移。要求部件 mobility 为 MOVABLE。

用法（在 SpaceMind_UE5 目录下运行）:
    python environments/satellite_pipeline/target_motion.py --satellite CAPSTONE --rate_dps 0.5
    python environments/satellite_pipeline/target_motion.py --satellite CAPSTONE --rate_dps 2.0 --axis tumble
    python environments/satellite_pipeline/target_motion.py --satellite CAPSTONE --reset_only   # 只复位姿态

部件 actor 名从 _tmp/session_<卫星>/name_label_mapping.txt 读取（ue_daemon 换星时导出）。
持续运行直到被外部终止（run_phase_e.py 负责启停），退出前不复位；
每个 run 开始前由 runner 先执行一次 --reset_only 保证初始姿态一致。
"""
import argparse
import math
import sys
import time
from pathlib import Path

import airsim

TMP_DIR = Path(__file__).resolve().parent / "_tmp"
ENV_ACTORS = {"earth", "moon", "MilkyWay"}  # mapping 里的非卫星 actor


def load_part_actors(satellite: str, mapping_file: str | None) -> list[str]:
    """从 name_label_mapping.txt 读取属于目标卫星的部件 actor 名。"""
    path = Path(mapping_file) if mapping_file else TMP_DIR / f"session_{satellite}" / "name_label_mapping.txt"
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {path} (需要 ue_daemon 换星时导出)")
    actors = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "|" not in line:
            continue
        name, label = line.split("|", 1)
        if label in ENV_ACTORS:
            continue
        if label.startswith(f"{satellite}_") or label.split("_")[0] == satellite:
            actors.append(name)
    if not actors:
        raise RuntimeError(f"No part actors for '{satellite}' in {path}")
    return actors


def quat_mul(a: airsim.Quaternionr, b: airsim.Quaternionr) -> airsim.Quaternionr:
    """四元数乘法 a*b（先施加 b 再施加 a）。"""
    return airsim.Quaternionr(
        x_val=a.w_val * b.x_val + a.x_val * b.w_val + a.y_val * b.z_val - a.z_val * b.y_val,
        y_val=a.w_val * b.y_val - a.x_val * b.z_val + a.y_val * b.w_val + a.z_val * b.x_val,
        z_val=a.w_val * b.z_val + a.x_val * b.y_val - a.y_val * b.x_val + a.z_val * b.w_val,
        w_val=a.w_val * b.w_val - a.x_val * b.x_val - a.y_val * b.y_val - a.z_val * b.z_val,
    )


def axis_angle_quat(axis: tuple[float, float, float], angle_rad: float) -> airsim.Quaternionr:
    s = math.sin(angle_rad / 2.0)
    return airsim.Quaternionr(
        x_val=axis[0] * s, y_val=axis[1] * s, z_val=axis[2] * s, w_val=math.cos(angle_rad / 2.0)
    )


AXES = {
    "yaw": (0.0, 0.0, 1.0),                     # 绕世界 Z 轴自旋（NED，Z 向下）
    "pitch": (0.0, 1.0, 0.0),
    "tumble": tuple(v / math.sqrt(3) for v in (1.0, 1.0, 1.0)),  # 斜轴翻滚
}


def main():
    parser = argparse.ArgumentParser(description="Rotate target satellite via AirSim simSetObjectPose")
    parser.add_argument("--satellite", required=True)
    parser.add_argument("--rate_dps", type=float, default=0.5, help="角速率 度/秒")
    parser.add_argument("--axis", choices=list(AXES.keys()), default="yaw")
    parser.add_argument("--hz", type=float, default=5.0, help="姿态更新频率")
    parser.add_argument("--mapping", default=None, help="name_label_mapping.txt 路径，默认从 _tmp/session_* 找")
    parser.add_argument("--reset_only", action="store_true", help="只把姿态复位为初始朝向后退出")
    args = parser.parse_args()

    actors = load_part_actors(args.satellite, args.mapping)
    print(f"[target_motion] {args.satellite}: {len(actors)} part actors", flush=True)

    client = airsim.VehicleClient()
    client.confirmConnection()

    # 读取各部件当前位姿；位置保持不变，姿态统一从 identity（放置时 BASE_ROTATION=0）开始
    identity = airsim.Quaternionr(0.0, 0.0, 0.0, 1.0)
    positions = {}
    for name in actors:
        pose = client.simGetObjectPose(name)
        if pose is None or math.isnan(pose.position.x_val):
            print(f"[target_motion] ERROR: simGetObjectPose failed for '{name}'", flush=True)
            sys.exit(2)
        positions[name] = pose.position

    def apply_orientation(q):
        # 注意：UE 在"新位姿与当前位姿相同"时 SetActorLocationAndRotation 返回 False，
        # 所以不能用返回值判断成败，以读回姿态校验为准
        for name in actors:
            client.simSetObjectPose(name, airsim.Pose(positions[name], q), True)

    def verify_orientation(q, tol=1e-3) -> bool:
        for name in actors:
            o = client.simGetObjectPose(name).orientation
            dot = abs(o.x_val * q.x_val + o.y_val * q.y_val + o.z_val * q.z_val + o.w_val * q.w_val)
            if not (1.0 - tol <= dot <= 1.0 + tol):
                print(f"[target_motion] ERROR: '{name}' orientation mismatch (dot={dot:.4f}), "
                      f"actor 可能不是 MOVABLE", flush=True)
                return False
        return True

    apply_orientation(identity)
    time.sleep(0.3)
    if not verify_orientation(identity):
        sys.exit(2)
    print("[target_motion] Orientation reset to identity", flush=True)

    if args.reset_only:
        return

    axis = AXES[args.axis]
    rate_rad = math.radians(args.rate_dps)
    interval = 1.0 / max(args.hz, 0.5)
    t0 = time.time()
    last_report = t0
    print(f"[target_motion] Spinning: axis={args.axis} rate={args.rate_dps} deg/s", flush=True)

    while True:
        now = time.time()
        angle = rate_rad * (now - t0)
        apply_orientation(quat_mul(axis_angle_quat(axis, angle), identity))
        if now - last_report >= 60:
            print(f"[target_motion] t={now - t0:.0f}s angle={math.degrees(angle) % 360:.1f} deg", flush=True)
            last_report = now
        time.sleep(interval)


if __name__ == "__main__":
    main()
