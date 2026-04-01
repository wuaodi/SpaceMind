import base64
import json
import logging
import math
import sys
from io import BytesIO
from typing import Any, Optional

import redis
from PIL import Image

from ..redis_contract import (
    KEY_LATEST_IMAGE,
    KEY_LATEST_LIDAR,
    KEY_LATEST_POSE_TRUTH,
    KEY_LATEST_SEGMENTATION,
    SENSOR_CACHE_KEYS,
    TOPIC_EXPOSURE,
    TOPIC_IMAGE,
    TOPIC_POSE,
    TOPIC_POSE_CHANGE,
)

logger = logging.getLogger("spacemind.mcp")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _redis_client() -> redis.Redis:
    return redis.Redis(host="localhost", port=6379, db=0, socket_timeout=2.0)


def read_json_snapshot(key: str) -> Optional[dict[str, Any]]:
    client = _redis_client()
    try:
        client.ping()
        raw = client.get(key)
    finally:
        client.close()
    if not raw:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    data = json.loads(raw)
    return data if isinstance(data, dict) else None


def publish_json_message(channel: str, payload: dict[str, Any]) -> None:
    client = _redis_client()
    try:
        client.ping()
        client.publish(channel, json.dumps(payload))
    finally:
        client.close()


def decode_base64_image(image_b64: str) -> Image.Image:
    image = Image.open(BytesIO(base64.b64decode(image_b64)))
    image.load()
    return image


def encode_png_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_snapshot_image(key: str = KEY_LATEST_IMAGE) -> Optional[tuple[dict[str, Any], Image.Image]]:
    payload = read_json_snapshot(key)
    if not payload:
        return None
    image_b64 = payload.get("data", "")
    if not image_b64:
        return None
    return payload, decode_base64_image(image_b64)


def compute_lidar_surface_distance(lidar_snapshot: Optional[dict[str, Any]]) -> Optional[float]:
    """Estimate target surface distance from the latest lidar snapshot."""
    if not isinstance(lidar_snapshot, dict):
        return None
    points = lidar_snapshot.get("points", [])
    if not isinstance(points, list) or len(points) < 3:
        return None

    nearest_distance: Optional[float] = None
    usable_count = len(points) - (len(points) % 3)
    for idx in range(0, usable_count, 3):
        try:
            x = float(points[idx])
            y = float(points[idx + 1])
            z = float(points[idx + 2])
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (x, y, z)):
            continue
        distance = math.sqrt(x * x + y * y + z * z)
        if distance <= 0:
            continue
        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance

    return round(nearest_distance, 3) if nearest_distance is not None else None


__all__ = [
    "KEY_LATEST_IMAGE",
    "KEY_LATEST_LIDAR",
    "KEY_LATEST_POSE_TRUTH",
    "KEY_LATEST_SEGMENTATION",
    "SENSOR_CACHE_KEYS",
    "TOPIC_EXPOSURE",
    "TOPIC_IMAGE",
    "TOPIC_POSE",
    "TOPIC_POSE_CHANGE",
    "_redis_client",
    "compute_lidar_surface_distance",
    "decode_base64_image",
    "encode_png_base64",
    "load_snapshot_image",
    "logger",
    "publish_json_message",
    "read_json_snapshot",
]
