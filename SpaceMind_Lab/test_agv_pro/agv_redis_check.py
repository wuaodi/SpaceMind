#!/usr/bin/env python3

# myAGV Pro Redis 传感器结果检查脚本
#
# 作用:
#   检查 agv_ros_sensor_bridge.py 是否已经把图像和雷达数据成功写入 Redis。
#   这个脚本本身不会驱动小车，只负责读 Redis 并打印摘要。
#
# 常用命令:
#   python test_agv_pro/agv_redis_check.py
#     默认检查:
#       agv_test:latest_image_data
#       agv_test:latest_lidar_data
#
#   python test_agv_pro/agv_redis_check.py --timeout 10
#     等待最多 10 秒，适合桥接刚启动时使用。
#
#   python test_agv_pro/agv_redis_check.py --redis-prefix ""
#     检查不带前缀的正式 Redis key。
#
# 运行效果:
#   1. 如果两个 key 都存在，会打印图像时间戳、尺寸和雷达点数
#   2. 如果雷达点存在，还会打印简单的 x/y 范围
#   3. 如果超时还没等到数据，会退出并给出错误提示

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import redis


def _print(message: str) -> None:
    print(f"[agv_redis_check] {message}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect Redis keys written by agv_ros_sensor_bridge.")
    parser.add_argument("--redis-host", default="127.0.0.1", help="Redis host.")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port.")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis database index.")
    parser.add_argument("--redis-prefix", default="agv_test:", help="Prefix applied to Redis keys.")
    parser.add_argument("--timeout", type=float, default=8.0, help="How long to wait for keys.")
    parser.add_argument("--poll-seconds", type=float, default=0.2, help="Polling interval.")
    return parser


def _read_json(client: redis.Redis, key: str) -> dict[str, Any] | None:
    raw = client.get(key)
    if not raw:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    data = json.loads(raw)
    return data if isinstance(data, dict) else None


def main() -> int:
    args = _build_parser().parse_args()
    image_key = f"{args.redis_prefix}latest_image_data"
    lidar_key = f"{args.redis_prefix}latest_lidar_data"

    try:
        client = redis.Redis(host=args.redis_host, port=args.redis_port, db=args.redis_db, socket_timeout=2.0)
        client.ping()
    except Exception as exc:
        _print(f"Redis connection failed: {exc}")
        return 1

    deadline = time.time() + args.timeout
    image = None
    lidar = None
    while time.time() < deadline:
        image = _read_json(client, image_key)
        lidar = _read_json(client, lidar_key)
        if image and lidar:
            break
        time.sleep(max(args.poll_seconds, 0.05))

    if not image or not lidar:
        _print(f"Timed out waiting for keys {image_key!r} and {lidar_key!r}")
        return 1

    width = image.get("width")
    height = image.get("height")
    timestamp = image.get("timestamp")
    total_points = int(lidar.get("total_points", 0) or 0)
    _print(f"Image ok: timestamp={timestamp}, size={width}x{height}, key={image_key}")
    _print(f"LiDAR ok: total_points={total_points}, key={lidar_key}")

    points = lidar.get("points", [])
    if isinstance(points, list) and len(points) >= 3:
        xs = points[0::3]
        ys = points[1::3]
        _print(
            "LiDAR bounds: "
            f"x=[{min(xs):.3f}, {max(xs):.3f}], "
            f"y=[{min(ys):.3f}, {max(ys):.3f}]"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
