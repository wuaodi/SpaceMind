"""Shared Redis topics and snapshot keys for the SpaceMind runtime."""

TOPIC_POSE = "topic.pose"
TOPIC_POSE_CHANGE = "topic.pose_change"
TOPIC_EXPOSURE = "topic.exposure"
TOPIC_IMAGE = "topic.img"

KEY_LATEST_IMAGE = "latest_image_data"
KEY_LATEST_SEGMENTATION = "latest_segmentation_data"
KEY_LATEST_LIDAR = "latest_lidar_data"
KEY_LATEST_POSE_TRUTH = "latest_pose_truth"

SENSOR_CACHE_KEYS = (
    KEY_LATEST_IMAGE,
    KEY_LATEST_SEGMENTATION,
    KEY_LATEST_LIDAR,
    KEY_LATEST_POSE_TRUTH,
)
