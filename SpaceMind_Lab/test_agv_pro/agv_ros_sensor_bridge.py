#!/usr/bin/env python3

# myAGV Pro ROS2 传感器到 Redis 的桥接脚本
#
# 作用:
#   订阅小车上的 ROS2 相机与雷达话题，把数据转换成当前 SpaceMind
#   传感器工具可以理解的 Redis 结构，便于后续接入主系统。
#
# 默认桥接:
#   图像话题: /camera/color/image_raw
#   雷达话题: /scan
#
# 默认写入的 Redis key:
#   agv_test:latest_image_data
#   agv_test:latest_lidar_data
#
# 图像写入内容:
#   {name, timestamp, width, height, data}
#   其中 data 是 PNG 的 base64 字符串
#
# 雷达写入内容:
#   {timestamp, points, total_points, angle_min, angle_max, angle_increment, range_min, range_max}
#   其中 points 是扁平 XYZ 数组，z 固定为 0.0
#
# 常用命令:
#   python test_agv_pro/agv_ros_sensor_bridge.py
#     持续监听默认图像和雷达话题，并把结果写到 agv_test: 前缀的 Redis key。
#
#   python test_agv_pro/agv_ros_sensor_bridge.py --once
#     默认在图像和雷达都收到一帧后退出；若同时加 --image-only，则收到一帧图像后退出。
#
#   python test_agv_pro/agv_ros_sensor_bridge.py --image-topic /camera/color/image_raw --scan-topic /scan
#     显式指定话题，适合不同车或不同 launch 配置下调试。
#
# 运行效果:
#   1. 收到图像后，会更新 agv_test:latest_image_data，并发布 topic.img
#   2. 收到雷达后，会更新 agv_test:latest_lidar_data
#   3. 终端会打印每次更新的时间戳、图像尺寸和雷达点数
#
# 说明:
#   当前只桥接图像和 2D 雷达，不发布 latest_pose_truth。
#   这是因为真机没有与仿真环境完全等价的目标相对真值位姿。

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import redis
from PIL import Image as PILImage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.redis_contract import KEY_LATEST_IMAGE, KEY_LATEST_LIDAR, TOPIC_IMAGE


def _print(message: str) -> None:
    print(f"[agv_ros_sensor_bridge] {message}", flush=True)


def _make_timestamp_ns(sec: int, nanosec: int) -> str:
    timestamp_ns = int(sec) * 1_000_000_000 + int(nanosec)
    if timestamp_ns <= 0:
        timestamp_ns = time.time_ns()
    return str(timestamp_ns)


def _prefixed_key(prefix: str, base_key: str) -> str:
    return f"{prefix}{base_key}" if prefix else base_key


def _image_to_pil(msg: Any) -> PILImage:
    encoding = str(getattr(msg, "encoding", "") or "").lower()
    width = int(msg.width)
    height = int(msg.height)
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding == "rgb8":
        return PILImage.fromarray(data.reshape(height, width, 3), mode="RGB")
    if encoding == "bgr8":
        array = data.reshape(height, width, 3)[:, :, ::-1]
        return PILImage.fromarray(array, mode="RGB")
    if encoding == "mono8":
        return PILImage.fromarray(data.reshape(height, width), mode="L")
    if encoding == "rgba8":
        return PILImage.fromarray(data.reshape(height, width, 4), mode="RGBA").convert("RGB")
    if encoding == "bgra8":
        array = data.reshape(height, width, 4)[:, :, [2, 1, 0, 3]]
        return PILImage.fromarray(array, mode="RGBA").convert("RGB")
    raise ValueError(f"Unsupported image encoding: {encoding!r}")


def _encode_png_base64(image: PILImage) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class SensorBridgeNode:
    def __init__(
        self,
        node: Any,
        redis_client: redis.Redis,
        redis_prefix: str,
        image_topic: str,
        scan_topic: str,
        image_only: bool,
        once: bool,
    ) -> None:
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, LaserScan

        self.node = node
        self.redis = redis_client
        self.redis_prefix = redis_prefix
        self.image_only = image_only
        self.once = once
        self.received_image = False
        self.received_scan = False

        self.node.create_subscription(Image, image_topic, self._on_image, qos_profile_sensor_data)
        _print(f"Subscribed image topic: {image_topic}")
        if self.image_only:
            _print("Image-only mode enabled; LaserScan subscription disabled.")
        else:
            self.node.create_subscription(LaserScan, scan_topic, self._on_scan, qos_profile_sensor_data)
            _print(f"Subscribed scan topic: {scan_topic}")

    def _on_image(self, msg: Any) -> None:
        try:
            image = _image_to_pil(msg)
            timestamp = _make_timestamp_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
            payload = {
                "name": f"agv_color_{timestamp}.png",
                "timestamp": timestamp,
                "width": image.width,
                "height": image.height,
                "data": _encode_png_base64(image),
            }
            payload_json = json.dumps(payload)
            key = _prefixed_key(self.redis_prefix, KEY_LATEST_IMAGE)
            self.redis.set(key, payload_json)
            self.redis.publish(TOPIC_IMAGE, payload_json)
            self.received_image = True
            _print(f"Updated {key} and published {TOPIC_IMAGE}: {payload['width']}x{payload['height']} @ {timestamp}")
            self._maybe_finish()
        except Exception as exc:
            _print(f"Image callback failed: {exc}")

    def _on_scan(self, msg: Any) -> None:
        try:
            points: list[float] = []
            for index, value in enumerate(msg.ranges):
                if not math.isfinite(value) or value < msg.range_min or value > msg.range_max:
                    continue
                angle = msg.angle_min + index * msg.angle_increment
                points.extend(
                    [
                        round(value * math.cos(angle), 4),
                        round(value * math.sin(angle), 4),
                        0.0,
                    ]
                )
            timestamp = _make_timestamp_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
            payload = {
                "timestamp": timestamp,
                "points": points,
                "total_points": len(points) // 3,
                "angle_min": msg.angle_min,
                "angle_max": msg.angle_max,
                "angle_increment": msg.angle_increment,
                "range_min": msg.range_min,
                "range_max": msg.range_max,
            }
            key = _prefixed_key(self.redis_prefix, KEY_LATEST_LIDAR)
            self.redis.set(key, json.dumps(payload))
            self.received_scan = True
            _print(f"Updated {key}: total_points={payload['total_points']} @ {timestamp}")
            self._maybe_finish()
        except Exception as exc:
            _print(f"Scan callback failed: {exc}")

    def _maybe_finish(self) -> None:
        if not self.once:
            return
        if self.image_only and self.received_image:
            self.node.get_logger().info("Received one image in image-only mode; shutting down.")
            import rclpy

            rclpy.shutdown()
            return
        if self.received_image and self.received_scan:
            self.node.get_logger().info("Received both image and scan once; shutting down.")
            import rclpy

            rclpy.shutdown()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge myAGV Pro ROS2 image/scan topics into Redis.")
    parser.add_argument("--image-topic", default="/camera/color/image_raw", help="ROS2 image topic.")
    parser.add_argument("--scan-topic", default="/scan", help="ROS2 LaserScan topic.")
    parser.add_argument("--redis-host", default="127.0.0.1", help="Redis host.")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port.")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis database index.")
    parser.add_argument("--redis-prefix", default="agv_test:", help="Prefix applied to Redis keys.")
    parser.add_argument("--image-only", action="store_true", help="Bridge only image frames and skip LaserScan subscription.")
    parser.add_argument("--once", action="store_true", help="Exit after one image in --image-only mode, otherwise after one image and one scan.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    try:
        import rclpy
        from rclpy.node import Node
    except Exception as exc:  # pragma: no cover - depends on ROS target
        _print(f"Failed to import ROS2 Python packages: {exc}")
        return 1

    try:
        redis_client = redis.Redis(host=args.redis_host, port=args.redis_port, db=args.redis_db, socket_timeout=2.0)
        redis_client.ping()
    except Exception as exc:
        _print(f"Redis connection failed: {exc}")
        return 1

    try:
        rclpy.init()
        node = Node("agv_ros_sensor_bridge")
        SensorBridgeNode(
            node=node,
            redis_client=redis_client,
            redis_prefix=args.redis_prefix,
            image_topic=args.image_topic,
            scan_topic=args.scan_topic,
            image_only=args.image_only,
            once=args.once,
        )
        _print(f"Writing Redis keys with prefix {args.redis_prefix!r}")
        rclpy.spin(node)
        node.destroy_node()
        return 0
    except KeyboardInterrupt:
        _print("Interrupted by user.")
        return 130
    except Exception as exc:  # pragma: no cover - hardware dependent
        _print(f"Bridge failed: {exc}")
        return 1
    finally:
        try:
            redis_client.close()
        except Exception:
            pass
        try:
            import rclpy

            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
