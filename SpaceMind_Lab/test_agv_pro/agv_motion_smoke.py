#!/usr/bin/env python3

# myAGV Pro 直连底盘基础动作测试
#
# 作用:
#   不经过 Redis，直接通过 pymycobot 连接小车控制器，执行一个单独动作。
#   适合先确认串口、底盘、急停状态和基础运动是否正常。
#
# 常用命令:
#   python test_agv_pro/agv_motion_smoke.py
#     默认让小车以较低速度前进约 1 秒，然后自动 stop。
#
#   python test_agv_pro/agv_motion_smoke.py --action backward --speed 0.3 --duration 1.0
#     小车后退约 1 秒，然后自动 stop。
#
#   python test_agv_pro/agv_motion_smoke.py --action left --speed 0.2 --duration 1.0
#     按当前真机实测映射，小车向左平移约 1 秒，然后自动 stop。
#
#   python test_agv_pro/agv_motion_smoke.py --action right --speed 0.2 --duration 1.0
#     按当前真机实测映射，小车向右平移约 1 秒，然后自动 stop。
#
#   python test_agv_pro/agv_motion_smoke.py --action turn_left --speed 0.2 --duration 1.0
#     按当前真机实测映射，小车原地左转约 1 秒，然后自动 stop。
#
#   python test_agv_pro/agv_motion_smoke.py --action turn_right --speed 0.2 --duration 1.0
#     按当前真机实测映射，小车原地右转约 1 秒，然后自动 stop。
#
#   python test_agv_pro/agv_motion_smoke.py --action stop
#     直接发送 stop，不执行其他动作。
#
# 说明:
#   1. 默认串口是 /dev/agvpro_controller
#   2. 默认波特率是 1000000
#   3. 脚本结束、报错或中断时都会尽量调用 stop()，防止小车持续运动

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Callable


def _print(message: str) -> None:
    print(f"[agv_motion_smoke] {message}", flush=True)


def _safe_call(obj: Any, method_name: str) -> Any:
    method = getattr(obj, method_name, None)
    if not callable(method):
        return "not available"
    try:
        return method()
    except Exception as exc:  # pragma: no cover - hardware dependent
        return f"error: {exc}"


def _resolve_method(obj: Any, *candidates: str) -> Callable[[float], Any]:
    for name in candidates:
        method = getattr(obj, name, None)
        if callable(method):
            return method
    raise AttributeError(f"No supported method found among: {', '.join(candidates)}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a direct myAGV Pro motion smoke test.")
    parser.add_argument("--port", default="/dev/agvpro_controller", help="Controller serial device.")
    parser.add_argument("--baudrate", type=int, default=1_000_000, help="Controller baudrate.")
    parser.add_argument("--speed", type=float, default=0.3, help="Command speed parameter.")
    parser.add_argument("--duration", type=float, default=1.0, help="How long to run the motion command.")
    parser.add_argument(
        "--action",
        choices=["forward", "backward", "left", "right", "turn_left", "turn_right", "stop"],
        default="forward",
        help="Single action to execute.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable pymycobot debug logging.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        from pymycobot import MyAGVPro
    except Exception as exc:  # pragma: no cover - import depends on target machine
        _print(f"Failed to import pymycobot.MyAGVPro: {exc}")
        return 1

    action_to_method: dict[str, tuple[str, ...]] = {
        "forward": ("move_forward",),
        "backward": ("move_backward",),
        "left": ("move_right_lateral", "move_right"),
        "right": ("move_left_lateral", "move_left"),
        "turn_left": ("turn_right",),
        "turn_right": ("turn_left",),
    }

    agv = None
    try:
        _print(f"Connecting to {args.port} @ {args.baudrate} ...")
        agv = MyAGVPro(args.port, baudrate=args.baudrate, debug=args.debug)
        agv.power_on()
        _print("power_on() sent.")

        status_items = {
            "get_estop_state": _safe_call(agv, "get_estop_state"),
            "get_motor_status": _safe_call(agv, "get_motor_status"),
            "get_auto_report_message": _safe_call(agv, "get_auto_report_message"),
        }
        for key, value in status_items.items():
            _print(f"{key}: {value}")

        if args.action == "stop":
            agv.stop()
            _print("stop() sent.")
            return 0

        method_names = action_to_method[args.action]
        method = _resolve_method(agv, *method_names)
        _print(f"Running {method.__name__} speed={args.speed} for {args.duration:.2f}s")
        method(args.speed)
        time.sleep(max(args.duration, 0.0))
        agv.stop()
        _print("Motion complete. stop() sent.")
        return 0
    except KeyboardInterrupt:
        _print("Interrupted by user.")
        return 130
    except Exception as exc:  # pragma: no cover - hardware dependent
        _print(f"Smoke test failed: {exc}")
        return 1
    finally:
        if agv is not None:
            try:
                agv.stop()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
