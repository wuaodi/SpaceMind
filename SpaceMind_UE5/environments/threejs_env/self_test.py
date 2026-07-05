# 启动:
#   python self_test.py --scenario smoke
#   python self_test.py --scenario move
#   python self_test.py --scenario exposure
#
# 作用:
#   在 threejs_env 已启动的前提下，直接通过 Redis 做最小链路验证。

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import redis


def print_line(message: str) -> None:
    print(f"[threejs_env self_test] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-test the Three.js environment via Redis.")
    parser.add_argument(
        "--scenario",
        choices=["smoke", "move", "exposure"],
        default="smoke",
        help="Which self-test scenario to run.",
    )
    return parser.parse_args()


def redis_client() -> redis.Redis:
    return redis.Redis(host="127.0.0.1", port=6379, db=0, socket_timeout=2.0)


def get_json(r: redis.Redis, key: str) -> dict[str, Any] | None:
    raw = r.get(key)
    if not raw:
        return None
    return json.loads(raw)


def wait_for_image_update(r: redis.Redis, previous_timestamp: str | None, timeout: float = 6.0) -> dict[str, Any] | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        image = get_json(r, "latest_image_data")
        if image and image.get("timestamp") != previous_timestamp:
            return image
        time.sleep(0.2)
    return None


def wait_for_pose(r: redis.Redis, timeout: float = 6.0) -> dict[str, Any] | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        pose = get_json(r, "latest_pose_truth")
        if pose:
            return pose
        time.sleep(0.2)
    return None


def publish_pose_change(r: redis.Redis, dx: float, dy: float, dz: float) -> None:
    payload = {
      "dx": dx,
      "dy": dy,
      "dz": dz,
      "dpitch": 0.0,
      "droll": 0.0,
      "dyaw": 0.0,
      "timestamp": str(int(time.time() * 1e9)),
    }
    r.publish("topic.pose_change", json.dumps(payload))


def run_smoke(r: redis.Redis) -> int:
    previous_image = get_json(r, "latest_image_data")
    previous_timestamp = previous_image.get("timestamp") if previous_image else None
    publish_pose_change(r, 0.0, 0.0, 0.0)
    pose = wait_for_pose(r)
    image = wait_for_image_update(r, previous_timestamp)
    if not pose or not image:
        print_line("Smoke test failed: 未同时拿到最新 pose 和 image。")
        return 1
    print_line(f"Smoke ok: image_timestamp={image['timestamp']}, distance={pose['distance']}, x={pose['relative_position']['x']}")
    return 0


def run_move(r: redis.Redis) -> int:
    pose_before = get_json(r, "latest_pose_truth")
    image_before = get_json(r, "latest_image_data")
    before_x = pose_before.get("relative_position", {}).get("x") if pose_before else None
    previous_timestamp = image_before.get("timestamp") if image_before else None
    publish_pose_change(r, 1.0, 0.0, 0.0)
    pose_after = wait_for_pose(r)
    image_after = wait_for_image_update(r, previous_timestamp)
    if not pose_after or not image_after:
        print_line("Move test failed: 未在位移后拿到新 pose/image。")
        return 1
    after_x = pose_after.get("relative_position", {}).get("x")
    print_line(f"Move ok: x_before={before_x}, x_after={after_x}, image_timestamp={image_after['timestamp']}")
    return 0


def run_exposure(r: redis.Redis) -> int:
    image_before = get_json(r, "latest_image_data")
    previous_timestamp = image_before.get("timestamp") if image_before else None
    r.publish("topic.exposure", json.dumps({"exposure_value": 1.5}))
    image_after = wait_for_image_update(r, previous_timestamp, timeout=8.0)
    if not image_after:
        print_line("Exposure test failed: 曝光命令后没有等到新的 image timestamp。")
        return 1
    print_line(f"Exposure ok: image_timestamp_before={previous_timestamp}, image_timestamp_after={image_after['timestamp']}")
    return 0


def main() -> int:
    args = parse_args()
    r = redis_client()
    try:
        if not r.ping():
            print_line("Redis ping failed.")
            return 1
    except Exception as exc:
        print_line(f"Redis 连接失败: {exc}")
        return 1

    if args.scenario == "smoke":
        return run_smoke(r)
    if args.scenario == "move":
        return run_move(r)
    if args.scenario == "exposure":
        return run_exposure(r)
    print_line(f"未知场景: {args.scenario}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
