#!/usr/bin/env python3

# myAGV Pro Redis 兼容运动执行器
#
# 作用:
#   订阅当前 SpaceMind 已在使用的 Redis 频道 topic.pose_change，
#   把消息里的 dx / dy / dyaw 转成小车的前进、横移和转向动作。
#   这样上游 MCP 工具层可以继续保持和以前一致的消息格式。
#
# 坐标兼容规则:
#   1. dx > 0: 小车前进
#   2. dx < 0: 小车后退
#   3. dy > 0: 小车右移
#   4. dy < 0: 小车左移
#   5. dyaw > 0: 小车右转
#   6. dyaw < 0: 小车左转
#   7. dz / dpitch / droll: 当前不支持，只打印 warning 并忽略
#
# 重要:
#   topic.pose_change 里的 dyaw / dpitch / droll 是弧度。
#   这与 tools/server_tools/env.py 的实现保持一致:
#   set_attitude() 对外接收角度，但写入 Redis 时会转成弧度。
#
# 常用命令:
#   python test_agv_pro/agv_pose_change_executor.py
#     启动执行器，持续监听 topic.pose_change。
#
#   python test_agv_pro/agv_pose_change_executor.py --once
#     只处理一条消息，执行完后退出。
#
#   python test_agv_pro/agv_pose_change_executor.py --linear-speed-mps 0.25 --angular-speed-degps 16
#     调低平移和转向速度，用更保守的方式测试。
#
# 配套测试命令:
#   python - <<'PY'
#   import json, time, redis
#   r = redis.Redis(host="127.0.0.1", port=6379, db=0)
#   r.publish("topic.pose_change", json.dumps({
#       "dx": 0.5,
#       "dy": 0.0,
#       "dz": 0.0,
#       "dpitch": 0.0,
#       "droll": 0.0,
#       "dyaw": 0.0,
#       "timestamp": str(time.time_ns()),
#   }))
#   PY
#     小车前进约 0.5 米对应的定时动作，然后 stop。
#
#   python - <<'PY'
#   import json, time, math, redis
#   r = redis.Redis(host="127.0.0.1", port=6379, db=0)
#   r.publish("topic.pose_change", json.dumps({
#       "dx": 0.0,
#       "dy": 0.3,
#       "dz": 0.0,
#       "dpitch": 0.0,
#       "droll": 0.0,
#       "dyaw": 0.0,
#       "timestamp": str(time.time_ns()),
#   }))
#   PY
#     按当前真机实测映射，小车右移约 0.3 米对应的定时动作，然后 stop。
#
#   python - <<'PY'
#   import json, time, math, redis
#   r = redis.Redis(host="127.0.0.1", port=6379, db=0)
#   r.publish("topic.pose_change", json.dumps({
#       "dx": 0.0,
#       "dy": 0.0,
#       "dz": 0.0,
#       "dpitch": 0.0,
#       "droll": 0.0,
#       "dyaw": math.radians(30),
#       "timestamp": str(time.time_ns()),
#   }))
#   PY
#     小车原地右转约 30 度对应的定时动作，然后 stop。
#
# 说明:
#   第一版使用“定时法”，不是里程计闭环，所以存在误差。
#   每个子动作执行完后都会 stop() 一次，异常或中断时也会尽量 stop()。
#   注意：根据当前真机实测，SDK 的左右横移命名与实际方向相反：
#   move_left_lateral / move_left 实际表现为右移，
#   move_right_lateral / move_right 实际表现为左移。
#   当前真机实测下，turn_left / turn_right 的实际方向也与命名相反，
#   并且默认角速度更接近约 16 deg/s，而不是 45 deg/s。

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import redis

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.redis_contract import TOPIC_POSE_CHANGE


def _print(message: str) -> None:
    print(f"[agv_pose_change_executor] {message}", flush=True)


def _safe_close(pubsub: redis.client.PubSub | None) -> None:
    if pubsub is None:
        return
    try:
        pubsub.close()
    except Exception:
        pass


def _resolve_method(obj: Any, *candidates: str) -> Any:
    for name in candidates:
        method = getattr(obj, name, None)
        if callable(method):
            return method
    raise AttributeError(f"No supported method found among: {', '.join(candidates)}")


class TimedPoseExecutor:
    def __init__(self, agv: Any, linear_speed_mps: float, angular_speed_degps: float, turn_command_speed: float, settle_seconds: float):
        if linear_speed_mps <= 0:
            raise ValueError("linear_speed_mps must be > 0")
        if angular_speed_degps <= 0:
            raise ValueError("angular_speed_degps must be > 0")
        if turn_command_speed <= 0:
            raise ValueError("turn_command_speed must be > 0")
        self.agv = agv
        self.linear_speed_mps = linear_speed_mps
        self.angular_speed_degps = angular_speed_degps
        self.turn_command_speed = turn_command_speed
        self.settle_seconds = max(settle_seconds, 0.0)

    def execute(self, payload: dict[str, Any]) -> None:
        dx = float(payload.get("dx", 0.0) or 0.0)
        dy = float(payload.get("dy", 0.0) or 0.0)
        dz = float(payload.get("dz", 0.0) or 0.0)
        dpitch = float(payload.get("dpitch", 0.0) or 0.0)
        droll = float(payload.get("droll", 0.0) or 0.0)
        dyaw_rad = float(payload.get("dyaw", 0.0) or 0.0)

        ignored_axes = []
        if abs(dz) > 1e-6:
            ignored_axes.append(f"dz={dz:.3f}")
        if abs(dpitch) > 1e-6:
            ignored_axes.append(f"dpitch={dpitch:.3f}rad")
        if abs(droll) > 1e-6:
            ignored_axes.append(f"droll={droll:.3f}rad")
        if ignored_axes:
            _print("Ignoring unsupported axes: " + ", ".join(ignored_axes))

        if abs(dx) < 1e-6 and abs(dy) < 1e-6 and abs(dyaw_rad) < 1e-6:
            self.agv.stop()
            _print("Received zero delta. stop() sent.")
            return

        self._run_linear(dx, positive_method="move_forward", negative_method="move_backward", axis_name="dx")
        self._run_linear(dy, positive_method="move_left_lateral", negative_method="move_right_lateral", axis_name="dy")
        self._run_yaw(dyaw_rad)

    def _run_linear(self, distance_m: float, positive_method: str, negative_method: str, axis_name: str) -> None:
        if abs(distance_m) < 1e-6:
            return
        method_name = positive_method if distance_m > 0 else negative_method
        duration = abs(distance_m) / self.linear_speed_mps
        _print(f"{axis_name}: {distance_m:.3f}m -> {method_name} speed={self.linear_speed_mps:.3f} for {duration:.2f}s")
        if method_name == "move_right_lateral":
            method = _resolve_method(self.agv, "move_right_lateral", "move_right")
        elif method_name == "move_left_lateral":
            method = _resolve_method(self.agv, "move_left_lateral", "move_left")
        else:
            method = _resolve_method(self.agv, method_name)
        method(self.linear_speed_mps)
        time.sleep(duration)
        self.agv.stop()
        if self.settle_seconds:
            time.sleep(self.settle_seconds)

    def _run_yaw(self, yaw_rad: float) -> None:
        if abs(yaw_rad) < 1e-6:
            return
        yaw_deg = math.degrees(yaw_rad)
        duration = abs(yaw_deg) / self.angular_speed_degps
        method_name = "turn_left" if yaw_deg > 0 else "turn_right"
        _print(
            f"dyaw: {yaw_rad:.3f}rad ({yaw_deg:.2f}deg) -> "
            f"{method_name} command_speed={self.turn_command_speed:.3f} for {duration:.2f}s"
        )
        method = _resolve_method(self.agv, method_name)
        method(self.turn_command_speed)
        time.sleep(duration)
        self.agv.stop()
        if self.settle_seconds:
            time.sleep(self.settle_seconds)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute SpaceMind pose_change commands on myAGV Pro.")
    parser.add_argument("--port", default="/dev/agvpro_controller", help="Controller serial device.")
    parser.add_argument("--baudrate", type=int, default=1_000_000, help="Controller baudrate.")
    parser.add_argument("--debug", action="store_true", help="Enable pymycobot debug logging.")
    parser.add_argument("--redis-host", default="127.0.0.1", help="Redis host.")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port.")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis database index.")
    parser.add_argument("--channel", default=TOPIC_POSE_CHANGE, help="Redis channel to subscribe to.")
    parser.add_argument("--linear-speed-mps", type=float, default=0.3, help="Effective linear speed used for timing.")
    parser.add_argument("--angular-speed-degps", type=float, default=16.0, help="Effective yaw speed used for timing.")
    parser.add_argument("--turn-command-speed", type=float, default=0.3, help="Command speed argument passed to turn_left/turn_right.")
    parser.add_argument("--settle-seconds", type=float, default=0.2, help="Pause after each sub-command.")
    parser.add_argument("--once", action="store_true", help="Exit after handling a single command message.")
    parser.add_argument(
        "--use-odom",
        action="store_true",
        help="Reserved for future odom closed-loop mode. Not implemented in this first version.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if args.use_odom:
        _print("--use-odom is reserved for a future closed-loop version and is not implemented yet.")
        return 2

    try:
        from pymycobot import MyAGVPro
    except Exception as exc:  # pragma: no cover - import depends on target machine
        _print(f"Failed to import pymycobot.MyAGVPro: {exc}")
        return 1

    agv = None
    pubsub = None
    try:
        agv = MyAGVPro(args.port, baudrate=args.baudrate, debug=args.debug)
        agv.power_on()
        executor = TimedPoseExecutor(
            agv=agv,
            linear_speed_mps=args.linear_speed_mps,
            angular_speed_degps=args.angular_speed_degps,
            turn_command_speed=args.turn_command_speed,
            settle_seconds=args.settle_seconds,
        )

        client = redis.Redis(host=args.redis_host, port=args.redis_port, db=args.redis_db, socket_timeout=2.0)
        client.ping()
        pubsub = client.pubsub()
        pubsub.subscribe(args.channel)
        _print(f"Subscribed to {args.channel} on redis://{args.redis_host}:{args.redis_port}/{args.redis_db}")

        while True:
            message = pubsub.get_message(timeout=1.0)
            if not message or message.get("type") != "message":
                continue

            raw = message.get("data")
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                _print(f"Ignored non-dict payload: {payload!r}")
                continue

            _print(f"Received payload: {payload}")
            try:
                executor.execute(payload)
            except Exception as exc:
                agv.stop()
                _print(f"Command execution failed: {exc}")
                if args.once:
                    return 1
                continue

            if args.once:
                return 0
    except KeyboardInterrupt:
        _print("Interrupted by user.")
        return 130
    except Exception as exc:  # pragma: no cover - hardware dependent
        _print(f"Executor failed: {exc}")
        return 1
    finally:
        if agv is not None:
            try:
                agv.stop()
            except Exception:
                pass
        _safe_close(pubsub)


if __name__ == "__main__":
    sys.exit(main())
