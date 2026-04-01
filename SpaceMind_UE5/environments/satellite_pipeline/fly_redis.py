# 飞行控制、位姿真值、传感器数据采集
"""
坐标系与姿态约定：
1) 世界坐标系 World：目标在原点，X前Y右Z下（右手系）
2) 本体坐标系 Body：随航天器姿态旋转，轴向与World同向
3) 相机坐标系 Camera：相机在本体+X方向1m处，轴向与Body一致

初始位置设置：
1) 通过 --init_x/--init_y/--init_z/--init_yaw 设置服务航天器的初始世界坐标与航向
2) fly_redis.py 在 takeoff 后只调用一次 simSetVehiclePose 来写入初始位姿
3) host 运行过程中不会自动把环境重置回这个初始位姿
4) 想复现实验初始状态时，最稳的方法是重启 fly_redis.py 并带相同参数

启动示例：
python fly_redis.py --init_x -11 --init_y 0 --init_z 0
python fly_redis.py --init_x -11 --init_y 0 --init_z 0 --init_yaw 1.57   # 目标不在视野内
python fly_redis.py --noise --init_x -11 --init_y 0 --init_z 0
"""
import argparse
import airsim
import numpy as np
import time
import os
import sys
import datetime
import cv2
from pathlib import Path
import redis
import json
import base64
import math
import random

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.redis_contract import (
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


def log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[fly_redis {timestamp}] {message}", flush=True)


def calculate_camera_pose(x, y, z, pitch, roll, yaw):
    """
    根据服务航天器的位姿计算相机的位姿
    相机安装在服务航天器质心前1m处
    """
    spacecraft_position = airsim.Vector3r(x, y, z)
    spacecraft_orientation = airsim.to_quaternion(pitch, roll, yaw)
    offset_body = airsim.Vector3r(1.0, 0, 0)

    qw = spacecraft_orientation.w_val
    qx = spacecraft_orientation.x_val
    qy = spacecraft_orientation.y_val
    qz = spacecraft_orientation.z_val

    offset_world = airsim.Vector3r(
        (1 - 2 * (qy**2 + qz**2)) * offset_body.x_val +
        (2 * (qx * qy - qw * qz)) * offset_body.y_val +
        (2 * (qx * qz + qw * qy)) * offset_body.z_val,
        (2 * (qx * qy + qw * qz)) * offset_body.x_val +
        (1 - 2 * (qx**2 + qz**2)) * offset_body.y_val +
        (2 * (qy * qz - qw * qx)) * offset_body.z_val,
        (2 * (qx * qz - qw * qy)) * offset_body.x_val +
        (2 * (qy * qz + qw * qx)) * offset_body.y_val +
        (1 - 2 * (qx**2 + qy**2)) * offset_body.z_val
    )

    camera_world_pos = airsim.Vector3r(
        spacecraft_position.x_val + offset_world.x_val,
        spacecraft_position.y_val + offset_world.y_val,
        spacecraft_position.z_val + offset_world.z_val
    )
    camera_orientation = spacecraft_orientation
    distance_to_origin = math.sqrt(camera_world_pos.x_val**2 + camera_world_pos.y_val**2 + camera_world_pos.z_val**2)
    return camera_world_pos, camera_orientation, distance_to_origin


class AirSimDataCollector:
    def __init__(self, init_x=-11, init_y=0, init_z=0, init_yaw=0.0, enable_noise=False, noise_pos=0.1, noise_att=0.02):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        log("AirSim connected, API control enabled")

        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.pose_topic = TOPIC_POSE
        self.pose_change_topic = TOPIC_POSE_CHANGE
        self.image_topic = TOPIC_IMAGE
        self.exposure_topic = TOPIC_EXPOSURE
        log("Redis connected: localhost:6379/0")
        self.redis_client.delete(*SENSOR_CACHE_KEYS)
        log("Cleared stale Redis sensor cache")

        self.current_x = init_x
        self.current_y = init_y
        self.current_z = init_z
        self.current_pitch = 0
        self.current_roll = 0
        self.current_yaw = init_yaw

        self.enable_noise = enable_noise
        self.noise_pos = noise_pos
        self.noise_att = noise_att

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.base_path = Path(f"D:/dataset/{timestamp}")
        self.folders = {
            "lidar": self.base_path / "lidar0/data",
            "cam0_scene": self.base_path / "cam0_Scene/data",
            "cam0_seg": self.base_path / "cam0_Seg/data"
        }
        for folder in self.folders.values():
            folder.mkdir(parents=True, exist_ok=True)
        log(f"Dataset output: {self.base_path}")

        self.image_requests = [
            airsim.ImageRequest("cam0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("cam0", airsim.ImageType.Segmentation, False, False)
        ]

    def normalize_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def get_pose_truth(self):
        """计算相机坐标系下目标的相对位姿（目标固定在原点）"""
        current_pose = self.client.simGetVehiclePose()
        service_position = current_pose.position
        service_orientation = current_pose.orientation

        relative_world_x = 0 - service_position.x_val
        relative_world_y = 0 - service_position.y_val
        relative_world_z = 0 - service_position.z_val

        w = service_orientation.w_val
        x = service_orientation.x_val
        y = service_orientation.y_val
        z = service_orientation.z_val

        rotation_matrix_inv = [
            [1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)],
            [2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x)],
            [2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)]
        ]

        relative_body_x = rotation_matrix_inv[0][0]*relative_world_x + rotation_matrix_inv[0][1]*relative_world_y + rotation_matrix_inv[0][2]*relative_world_z
        relative_body_y = rotation_matrix_inv[1][0]*relative_world_x + rotation_matrix_inv[1][1]*relative_world_y + rotation_matrix_inv[1][2]*relative_world_z
        relative_body_z = rotation_matrix_inv[2][0]*relative_world_x + rotation_matrix_inv[2][1]*relative_world_y + rotation_matrix_inv[2][2]*relative_world_z

        relative_camera_x = relative_body_x - 1.0
        relative_camera_y = relative_body_y
        relative_camera_z = relative_body_z

        distance = math.sqrt(relative_camera_x**2 + relative_camera_y**2 + relative_camera_z**2)
        azimuth = math.atan2(relative_camera_y, relative_camera_x) if relative_camera_x != 0 else 0
        elevation = math.atan2(-relative_camera_z, math.sqrt(relative_camera_x**2 + relative_camera_y**2)) if distance > 0 else 0

        return {
            'relative_position': {'x': round(relative_camera_x, 3), 'y': round(relative_camera_y, 3), 'z': round(relative_camera_z, 3)},
            'distance': round(distance, 3),
            'azimuth_deg': round(math.degrees(azimuth), 1),
            'elevation_deg': round(math.degrees(elevation), 1),
            'service_spacecraft_pose': {
                'position': {'x': round(service_position.x_val, 3), 'y': round(service_position.y_val, 3), 'z': round(service_position.z_val, 3)},
                'orientation': {'w': round(service_orientation.w_val, 4), 'x': round(service_orientation.x_val, 4), 'y': round(service_orientation.y_val, 4), 'z': round(service_orientation.z_val, 4)}
            }
        }

    def collect_data(self, timestamp):
        pose_truth_data = self.get_pose_truth()
        pose_truth_data['timestamp'] = timestamp
        self.redis_client.set(KEY_LATEST_POSE_TRUTH, json.dumps(pose_truth_data))
        relative_pos = pose_truth_data['relative_position']
        log(
            "Pose truth published: "
            f"dist={pose_truth_data['distance']:.2f}m, "
            f"x={relative_pos['x']:.2f}, y={relative_pos['y']:.2f}, z={relative_pos['z']:.2f}, "
            f"az={pose_truth_data['azimuth_deg']:.1f}deg, el={pose_truth_data['elevation_deg']:.1f}deg"
        )

        lidar_data = self.client.getLidarData()
        points = lidar_data.point_cloud

        if points:
            lidar_filename = f"{timestamp}.asc"
            lidar_path = self.folders["lidar"] / lidar_filename
            with open(lidar_path, 'w') as f:
                for i in range(0, len(points) - 2, 3):
                    x, y, z = points[i], points[i+1], points[i+2]
                    f.write(f"{x},{y},{z}\n")

            points_list = []
            for i in range(0, len(points) - 2, 3):
                points_list.extend([points[i], points[i+1], points[i+2]])

            if len(points_list) > 900:
                random.seed(42)
                indices = list(range(0, len(points_list), 3))
                selected_indices = random.sample(indices, 300)
                sampled_points = []
                for idx in selected_indices:
                    sampled_points.extend([points_list[idx], points_list[idx+1], points_list[idx+2]])
                points_list = sampled_points

            lidar_message = {'timestamp': timestamp, 'points': points_list, 'total_points': len(points) // 3}
            self.redis_client.set(KEY_LATEST_LIDAR, json.dumps(lidar_message))
            log(f"LiDAR published: {lidar_message['total_points']} points")
        else:
            self.redis_client.set(KEY_LATEST_LIDAR, json.dumps({'timestamp': timestamp, 'points': [], 'total_points': 0}))
            log("LiDAR published: 0 points")

        responses = self.client.simGetImages(self.image_requests)
        if responses:
            for response in responses:
                filename = f"{timestamp}.png"
                if response.image_type == airsim.ImageType.Segmentation:
                    img_path = self.folders["cam0_seg"] / filename
                    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                    img_seg = img1d.reshape(response.height, response.width, 3)
                    cv2.imwrite(str(img_path), img_seg)
                    _, seg_data = cv2.imencode('.png', img_seg)
                    self.redis_client.set(KEY_LATEST_SEGMENTATION, json.dumps({
                        'name': filename, 'timestamp': timestamp, 'width': response.width, 'height': response.height,
                        'data': base64.b64encode(seg_data).decode('utf-8')
                    }))
                    log(f"Segmentation published: {filename} ({response.width}x{response.height})")
                elif response.image_type == airsim.ImageType.Scene:
                    img_path = self.folders["cam0_scene"] / filename
                    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                    img_rgb = img1d.reshape(response.height, response.width, 3)
                    cv2.imwrite(str(img_path), img_rgb)
                    _, img_data = cv2.imencode('.png', img_rgb)
                    image_message = {'name': filename, 'timestamp': timestamp, 'width': response.width, 'height': response.height, 'data': base64.b64encode(img_data).decode('utf-8')}
                    self.redis_client.publish(self.image_topic, json.dumps(image_message))
                    self.redis_client.set(KEY_LATEST_IMAGE, json.dumps(image_message))
                    log(f"RGB image published: {filename} ({response.width}x{response.height})")

    def fly_control(self):
        log("Initializing flight control")
        self.client.takeoffAsync().join()
        self.client.simSetVehiclePose(
            airsim.Pose(airsim.Vector3r(self.current_x, self.current_y, self.current_z),
                       airsim.to_quaternion(pitch=self.current_pitch, roll=self.current_roll, yaw=self.current_yaw)),
            True
        )
        time.sleep(0.5)
        log(f"Initial pose set: x={self.current_x}, y={self.current_y}, z={self.current_z}, yaw={self.current_yaw:.4f}")
        try:
            exposure_val = getattr(self, '_init_exposure', 0)
            self.client.simRunConsoleCommand(f"r.ExposureOffset {exposure_val}")
            log(f"Exposure initialized to {exposure_val}")
        except Exception:
            log("Failed to initialize exposure")
        log("Initial capture skipped; waiting for host trigger")

        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(self.pose_topic, self.pose_change_topic, self.exposure_topic)
        log(f"Subscribed topics: {self.pose_topic}, {self.pose_change_topic}, {self.exposure_topic}")

        while True:
            message = pubsub.get_message(timeout=1.0)
            if message and message['type'] == 'message':
                try:
                    data_str = message['data'].decode('utf-8')
                    data_dict = json.loads(data_str)

                    current_pose = self.client.simGetVehiclePose()
                    current_position = current_pose.position
                    current_orientation = current_pose.orientation
                    current_euler = airsim.to_eularian_angles(current_orientation)
                    self.current_pitch, self.current_roll, self.current_yaw = current_euler
                    self.current_x = current_position.x_val
                    self.current_y = current_position.y_val
                    self.current_z = current_position.z_val

                    channel = message['channel'].decode('utf-8')
                    log(f"Command received on {channel}: {data_dict}")

                    if channel == self.pose_topic:
                        x = data_dict.get('x', self.current_x)
                        y = data_dict.get('y', self.current_y)
                        z = data_dict.get('z', self.current_z)
                        pitch = data_dict.get('pitch', self.current_pitch)
                        roll = data_dict.get('roll', self.current_roll)
                        yaw = data_dict.get('yaw', self.current_yaw)
                        self.current_x, self.current_y, self.current_z = x, y, z
                        self.current_pitch, self.current_roll, self.current_yaw = pitch, roll, yaw
                        new_position = airsim.Vector3r(x, y, z)
                        new_orientation = airsim.to_quaternion(pitch, roll, yaw)
                        log(
                            "Applying absolute pose: "
                            f"x={x:.3f}, y={y:.3f}, z={z:.3f}, "
                            f"pitch={pitch:.4f}, roll={roll:.4f}, yaw={yaw:.4f}"
                        )

                    elif channel == self.pose_change_topic:
                        dx = data_dict.get('dx', 0)
                        dy = data_dict.get('dy', 0)
                        dz = data_dict.get('dz', 0)
                        dpitch = data_dict.get('dpitch', 0)
                        droll = data_dict.get('droll', 0)
                        dyaw = data_dict.get('dyaw', 0)

                        if self.enable_noise:
                            dx += np.random.normal(0, self.noise_pos)
                            dy += np.random.normal(0, self.noise_pos)
                            dz += np.random.normal(0, self.noise_pos)
                            dpitch += np.random.normal(0, self.noise_att)
                            droll += np.random.normal(0, self.noise_att)
                            dyaw += np.random.normal(0, self.noise_att)

                        w = current_orientation.w_val
                        x = current_orientation.x_val
                        y = current_orientation.y_val
                        z = current_orientation.z_val
                        rotation_matrix = [
                            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
                        ]
                        world_delta_x = rotation_matrix[0][0]*dx + rotation_matrix[0][1]*dy + rotation_matrix[0][2]*dz
                        world_delta_y = rotation_matrix[1][0]*dx + rotation_matrix[1][1]*dy + rotation_matrix[1][2]*dz
                        world_delta_z = rotation_matrix[2][0]*dx + rotation_matrix[2][1]*dy + rotation_matrix[2][2]*dz

                        new_position = airsim.Vector3r(
                            current_position.x_val + world_delta_x,
                            current_position.y_val + world_delta_y,
                            current_position.z_val + world_delta_z
                        )

                        relative_quat = airsim.to_quaternion(dpitch, droll, dyaw)
                        q1, q2 = current_orientation, relative_quat
                        new_w = q1.w_val*q2.w_val - q1.x_val*q2.x_val - q1.y_val*q2.y_val - q1.z_val*q2.z_val
                        new_x = q1.w_val*q2.x_val + q1.x_val*q2.w_val + q1.y_val*q2.z_val - q1.z_val*q2.y_val
                        new_y = q1.w_val*q2.y_val - q1.x_val*q2.z_val + q1.y_val*q2.w_val + q1.z_val*q2.x_val
                        new_z = q1.w_val*q2.z_val + q1.x_val*q2.y_val - q1.y_val*q2.x_val + q1.z_val*q2.w_val
                        new_orientation = airsim.Quaternionr(new_x, new_y, new_z, new_w)

                        new_euler = airsim.to_eularian_angles(new_orientation)
                        self.current_pitch = self.normalize_angle(new_euler[0])
                        self.current_roll = self.normalize_angle(new_euler[1])
                        self.current_yaw = self.normalize_angle(new_euler[2])
                        self.current_x = new_position.x_val
                        self.current_y = new_position.y_val
                        self.current_z = new_position.z_val
                        camera_world_pos, _, distance_to_origin = calculate_camera_pose(
                            self.current_x, self.current_y, self.current_z,
                            self.current_pitch, self.current_roll, self.current_yaw
                        )
                        log(
                            "Applying pose delta: "
                            f"dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}, "
                            f"dpitch={dpitch:.4f}, droll={droll:.4f}, dyaw={dyaw:.4f}"
                        )
                        log(
                            "Updated camera pose: "
                            f"x={camera_world_pos.x_val:.3f}, y={camera_world_pos.y_val:.3f}, z={camera_world_pos.z_val:.3f}, "
                            f"distance_to_target={distance_to_origin:.2f}m"
                        )

                    elif channel == self.exposure_topic:
                        exposure_value = data_dict.get('exposure_value', 0.0)
                        try:
                            self.client.simRunConsoleCommand(f"r.ExposureOffset {exposure_value}")
                            log(f"Exposure set: {exposure_value}")
                        except Exception:
                            log(f"Exposure command failed: {exposure_value}")
                        continue

                    if channel in (self.pose_topic, self.pose_change_topic):
                        new_pose = airsim.Pose(new_position, new_orientation)
                        self.client.simSetVehiclePose(new_pose, True)
                        time.sleep(0.5)
                        timestamp = str(int(time.time() * 1e9))
                        self.collect_data(timestamp)

                except json.JSONDecodeError:
                    log("Received invalid JSON payload")
                except Exception as e:
                    log(f"Error while processing command: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise', action='store_true', help='启用执行误差模拟')
    parser.add_argument('--noise_pos', type=float, default=0.1, help='位置噪声标准差(米)')
    parser.add_argument('--noise_att', type=float, default=0.02, help='姿态噪声标准差(弧度)')
    parser.add_argument('--init_x', type=float, default=-11, help='初始世界坐标 x；目标固定在原点，负 x 表示从目标前方起飞')
    parser.add_argument('--init_y', type=float, default=0, help='初始世界坐标 y；正 y 会让目标更偏向画面左侧，负 y 更偏向右侧')
    parser.add_argument('--init_z', type=float, default=0, help='初始世界坐标 z；默认与目标等高')
    parser.add_argument('--init_yaw', type=float, default=0, help='初始航向角(弧度)；0=朝向目标，1.57≈90°偏转使目标不在视野内')
    parser.add_argument('--init_exposure', type=float, default=0, help='初始曝光偏移；负=欠曝，正=过曝，0=默认')
    args = parser.parse_args()

    collector = AirSimDataCollector(
        init_x=args.init_x, init_y=args.init_y, init_z=args.init_z,
        init_yaw=args.init_yaw,
        enable_noise=args.noise, noise_pos=args.noise_pos, noise_att=args.noise_att
    )
    collector._init_exposure = args.init_exposure
    collector.fly_control()
