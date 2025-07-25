# 飞行控制、位姿真值、传感器数据采集
import airsim
import numpy as np
import time
import os
import datetime
import cv2
from pathlib import Path
import redis
import json
import base64
import math


def calculate_camera_pose(x, y, z, pitch, roll, yaw):
    """
    根据服务航天器的位姿计算相机的位姿
    相机安装在服务航天器质心前1m处
    
    参数:
    x, y, z: 服务航天器质心在世界系下的位置 (米)
    pitch, roll, yaw: 服务航天器的姿态角 (弧度)
    
    返回:
    相机在世界系下的位置和姿态
    """
    
    # 创建服务航天器的位姿
    spacecraft_position = airsim.Vector3r(x, y, z)
    
    # 将欧拉角转换为四元数
    spacecraft_orientation = airsim.to_quaternion(pitch, roll, yaw)
    
    # 定义相机在航天器本体坐标系下的偏移向量 (前方1米)
    offset_body = airsim.Vector3r(1.0, 0, 0)
    
    # 将偏移向量从本体坐标系旋转到世界坐标系
    # 使用四元数旋转公式
    qw = spacecraft_orientation.w_val
    qx = spacecraft_orientation.x_val
    qy = spacecraft_orientation.y_val
    qz = spacecraft_orientation.z_val
    
    # 旋转矩阵应用到偏移向量
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
    
    # 计算相机在世界系下的位置
    camera_world_pos = airsim.Vector3r(
        spacecraft_position.x_val + offset_world.x_val,
        spacecraft_position.y_val + offset_world.y_val,
        spacecraft_position.z_val + offset_world.z_val
    )
    
    # 相机的姿态与航天器姿态一致
    camera_orientation = spacecraft_orientation
    
    # 计算相机到原点的距离
    distance_to_origin = math.sqrt(camera_world_pos.x_val**2 + camera_world_pos.y_val**2 + camera_world_pos.z_val**2)

    
    return camera_world_pos, camera_orientation, distance_to_origin

class AirSimDataCollector:
    def __init__(self):
        # 连接到AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # 连接到Redis
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.pose_topic = 'topic.pose'  # 位姿数据订阅主题
        self.pose_change_topic = 'topic.pose_change'  # 位姿变化订阅主题
        self.image_topic = 'topic.img'  # 图像数据发布主题
        
        # 添加曝光控制订阅
        self.exposure_topic = 'topic.exposure'
        
        # 设置初始位置(米),姿态(弧度制)
        self.current_x = -21  # 注意这个是Airsim世界系下服务航天器位置，相机安装在服务航天器质心前方1米处
        self.current_y = 5
        self.current_z = 5
        self.current_pitch = 0
        self.current_roll = 0
        self.current_yaw = 0
        
        # 创建基于时间戳的数据存储目录
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.base_path = Path(f"D:/dataset/{timestamp}")
        
        # 创建存储不同类型数据的目录
        self.folders = {
            "lidar": self.base_path / "lidar0/data",
            "cam0_scene": self.base_path / "cam0_Scene/data",
            "cam0_seg": self.base_path / "cam0_Seg/data"
        }
        
        # 创建目录
        for folder in self.folders.values():
            folder.mkdir(parents=True, exist_ok=True)
        
        
        # 定义相机请求
        self.image_requests = [
            airsim.ImageRequest("cam0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("cam0", airsim.ImageType.Segmentation, False, False)
        ]
    

    def normalize_angle(self, angle):
        """归一化角度到[-π, π]"""
        return (angle + math.pi) % (2 * math.pi) - math.pi
    

    def get_pose_truth(self):
        """获取位姿真值
        
        计算服务航天器相机坐标系下目标航天器的相对位姿
        注意：相机安装在服务航天器质心前方1米处
        
        Returns:
            dict: 包含相对位姿信息的字典
        """
        # 获取服务航天器的当前位姿（世界坐标系）
        current_pose = self.client.simGetVehiclePose()
        service_position = current_pose.position
        service_orientation = current_pose.orientation
        
        # 目标航天器在世界坐标系的位置（固定在原点）
        target_world_x = 0
        target_world_y = 0
        target_world_z = 0
        
        # 计算世界坐标系下的相对位置
        relative_world_x = target_world_x - service_position.x_val
        relative_world_y = target_world_y - service_position.y_val
        relative_world_z = target_world_z - service_position.z_val
        
        # 将相对位置从世界坐标系转换到服务航天器本体坐标系
        # 使用四元数的共轭来进行逆旋转
        w = service_orientation.w_val
        x = service_orientation.x_val
        y = service_orientation.y_val
        z = service_orientation.z_val
        
        # 构建旋转矩阵的转置（逆旋转）
        rotation_matrix_inv = [
            [1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)],
            [2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x)],
            [2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)]
        ]
        
        # 应用逆旋转矩阵，得到本体坐标系下的相对位置
        relative_body_x = rotation_matrix_inv[0][0]*relative_world_x + rotation_matrix_inv[0][1]*relative_world_y + rotation_matrix_inv[0][2]*relative_world_z
        relative_body_y = rotation_matrix_inv[1][0]*relative_world_x + rotation_matrix_inv[1][1]*relative_world_y + rotation_matrix_inv[1][2]*relative_world_z
        relative_body_z = rotation_matrix_inv[2][0]*relative_world_x + rotation_matrix_inv[2][1]*relative_world_y + rotation_matrix_inv[2][2]*relative_world_z
        
        # 转换到相机坐标系：相机位于航天器质心前方1米处
        # 相机坐标系下的目标位置 = 本体坐标系下的目标位置 - 相机在本体系中的位置
        camera_offset_x = 1.0  # 相机在本体系x轴前方1米
        relative_camera_x = relative_body_x - camera_offset_x
        relative_camera_y = relative_body_y  # y轴无偏移
        relative_camera_z = relative_body_z  # z轴无偏移

        # 计算相对距离和方向角（基于相机坐标系）
        distance = math.sqrt(relative_camera_x**2 + relative_camera_y**2 + relative_camera_z**2)
        
        # 计算方位角（水平面内的角度）
        azimuth = math.atan2(relative_camera_y, relative_camera_x) if relative_camera_x != 0 else 0
        
        # 计算俯仰角
        elevation = math.atan2(-relative_camera_z, math.sqrt(relative_camera_x**2 + relative_camera_y**2)) if distance > 0 else 0
        
        # 返回位姿真值数据
        return {
            'relative_position': {
                'x': round(relative_camera_x, 3),
                'y': round(relative_camera_y, 3),
                'z': round(relative_camera_z, 3)
            },
            'distance': round(distance, 3),
            'azimuth_deg': round(math.degrees(azimuth), 1),
            'elevation_deg': round(math.degrees(elevation), 1),
            'service_spacecraft_pose': {
                'position': {
                    'x': round(service_position.x_val, 3),
                    'y': round(service_position.y_val, 3),
                    'z': round(service_position.z_val, 3)
                },
                'orientation': {
                    'w': round(service_orientation.w_val, 4),
                    'x': round(service_orientation.x_val, 4),
                    'y': round(service_orientation.y_val, 4),
                    'z': round(service_orientation.z_val, 4)
                }
            }
        }
    

    def collect_data(self, timestamp):
        """在当前位置收集所有传感器数据"""
        # 获取位姿真值，通过current_pose = self.client.simGetVehiclePose()
        # 这个函数得到的是Airsim世界系下的服务航天器（无人机）的位姿，需要将其转到服务航天器本体系下目标航天器的位姿
        # 目标航天器是不动的，原点在Airsim世界系的原点
        # 服务航天器本体系下目标航天器的位姿为：x,y,z,pitch,roll,yaw，单位为米和弧度
        
        # 调用独立的位姿真值获取函数
        pose_truth_data = self.get_pose_truth()
        pose_truth_data['timestamp'] = timestamp # 添加时间戳，用于后续数据对齐
        
        # 保存位姿真值数据到Redis
        self.redis_client.set('latest_pose_truth', json.dumps(pose_truth_data))
        
        # 打印位姿信息
        relative_pos = pose_truth_data['relative_position']
        distance = pose_truth_data['distance']
        azimuth = pose_truth_data['azimuth_deg']
        elevation = pose_truth_data['elevation_deg']
        
        print(f"发布位姿真值: 距离={distance:.2f}m, x={relative_pos['x']:.2f}m, y={relative_pos['y']:.2f}m, z={relative_pos['z']:.2f}m, 方位角={azimuth:.1f}°, 俯仰角={elevation:.1f}°")
        
        # 获取激光雷达数据
        lidar_data = self.client.getLidarData()
        points = lidar_data.point_cloud
        
        print(f"LiDAR调试: 获取到 {len(points) if points else 0} 个原始点")
        
        if points:
            # 保存激光雷达点云到文件
            lidar_filename = f"{timestamp}.asc"
            lidar_path = self.folders["lidar"] / lidar_filename

            with open(lidar_path, 'w') as f:
                # 确保我们只处理完整的点云数据（每3个值为一组）
                for i in range(0, len(points) - 2, 3):
                    x, y, z = points[i], points[i+1], points[i+2]
                    f.write(f"{x},{y},{z}\n")
            
            # 发布激光雷达数据到Redis
            # 将点云数据转换为列表格式
            points_list = []
            for i in range(0, len(points) - 2, 3):
                points_list.extend([points[i], points[i+1], points[i+2]])
            
            # 如果大于300个点就随机采样300个点
            if len(points_list) > 900:  # 900 = 300个3D点 * 3个坐标
                import random
                random.seed(42)
                # 创建索引列表，每3个值为一组
                indices = list(range(0, len(points_list), 3))
                # 随机选择300个点的索引
                selected_indices = random.sample(indices, 300)
                # 提取选中的点
                sampled_points = []
                for idx in selected_indices:
                    sampled_points.extend([points_list[idx], points_list[idx+1], points_list[idx+2]])
                points_list = sampled_points
            
            lidar_message = {
                'timestamp': timestamp,
                'points': points_list,
                'total_points': len(points) // 3
            }
            
            # 保存最新的激光雷达数据供工具使用 - 使用json.dumps确保正确序列化
            self.redis_client.set('latest_lidar_data', json.dumps(lidar_message))
            print(f"发布激光雷达数据: {lidar_message['total_points']}个点")

        
        # 获取相机图像
        responses = self.client.simGetImages(self.image_requests)
        if responses:
            for i, response in enumerate(responses):
                img_type = response.image_type          
                filename = f"{timestamp}.png"
                
                # 发布分割图像
                if img_type == airsim.ImageType.Segmentation:
                    folder_key = "cam0_seg"
                    img_path = self.folders[folder_key] / filename
                    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                    img_seg = img1d.reshape(response.height, response.width, 3)
                    cv2.imwrite(str(img_path), img_seg)
                    
                    # 发布分割图像数据到Redis
                    try:
                        _, seg_data = cv2.imencode('.png', img_seg)
                        encoded_seg = base64.b64encode(seg_data).decode('utf-8')
                        
                        seg_message = {
                            'name': filename,
                            'timestamp': timestamp,
                            'width': response.width,
                            'height': response.height,
                            'data': encoded_seg
                        }
                        
                        # 保存最新的分割数据供工具使用
                        self.redis_client.set('latest_segmentation_data', json.dumps(seg_message))
                        print(f"发布分割图像: {filename}")
                        
                    except Exception as e:
                        print(f"发布分割图像时发生错误: {e}")
                
                # 发布RGB图像
                elif img_type == airsim.ImageType.Scene:
                    folder_key = "cam0_scene"
                    img_path = self.folders[folder_key] / filename
                    # 保存RGB图像
                    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                    img_rgb = img1d.reshape(response.height, response.width, 3)
                    cv2.imwrite(str(img_path), img_rgb)
                    
                    # 通过Redis发布RGB图像数据
                    try:
                        _, img_data = cv2.imencode('.png', img_rgb)  # 将图像编码为PNG格式
                        encoded_img = base64.b64encode(img_data).decode('utf-8')  # 转换为base64字符串
                        
                        image_message = {
                            'name': filename,
                            'timestamp': timestamp,
                            'width': response.width,
                            'height': response.height,
                            'data': encoded_img
                        }
                        
                        # redis pub 发布图像数据，数据驱动大模型分析
                        self.redis_client.publish(self.image_topic, str(image_message))
                        print(f"发布RGB图像: {filename} ({response.width}x{response.height})")
                        
                        # redis set 发布图像数据
                        self.redis_client.set('latest_image_data', json.dumps(image_message))
                        
                    except Exception as e:
                        print(f"发布图像数据时发生错误: {e}")
                
            
        
    
    def fly_control(self):
        """通过Redis订阅位姿命令控制服务航天器飞行并采集数据"""
        print("正在初始化服务航天器飞行和数据采集......")
        
        # 起飞
        self.client.takeoffAsync().join()

        # 初始服务航天器位置
        self.client.simSetVehiclePose(
            airsim.Pose(airsim.Vector3r(self.current_x, self.current_y, self.current_z), 
                       airsim.to_quaternion(pitch=self.current_pitch, roll=self.current_roll, yaw=self.current_yaw)), 
            True
        )
        time.sleep(0.5)  # 暂停0.5秒，确保服务航天器位姿设置到位了

        # 初始化成功信息
        print(f"🚀 初始化成功，初始位置为{self.current_x},{self.current_y},{self.current_z}，开始采集数据")

        
        # 采集初始位置的数据
        timestamp = str(int(time.time() * 1e9))
        self.collect_data(timestamp)

        
        # 创建Redis订阅对象，订阅三个主题
        # 订阅主题：位姿、位姿变化、曝光控制
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(self.pose_topic, self.pose_change_topic, self.exposure_topic)
        
        
        while True:
            # 持续监听订阅消息，设置超时以避免无限阻塞
            message = pubsub.get_message(timeout=1.0)
            
            if message and message['type'] == 'message':
                try:
                    print('='*50)
                    # 解析接收到的数据
                    data_str = message['data'].decode('utf-8')
                    data_dict = json.loads(data_str)

                    # 先获取当前真实绝对位姿（从AirSim获取，避免累加漂移）
                    current_pose = self.client.simGetVehiclePose()
                    current_position = current_pose.position
                    current_orientation = current_pose.orientation

                    # 转换为欧拉角（用于缓存和调试，可选）
                    current_euler = airsim.to_eularian_angles(current_orientation)  # 返回(pitch, roll, yaw)弧度
                    self.current_pitch, self.current_roll, self.current_yaw = current_euler
                    self.current_x = current_position.x_val
                    self.current_y = current_position.y_val
                    self.current_z = current_position.z_val
                    
                    # 判断是哪个主题的消息
                    if message['channel'].decode('utf-8') == self.pose_topic:
                        # 绝对位姿命令（直接设置）
                        # get方法的第二个参数是默认值，如果data_dict中没有这个键，则使用默认值
                        x = data_dict.get('x', self.current_x)
                        y = data_dict.get('y', self.current_y)
                        z = data_dict.get('z', self.current_z)
                        pitch = data_dict.get('pitch', self.current_pitch)
                        roll = data_dict.get('roll', self.current_roll)
                        yaw = data_dict.get('yaw', self.current_yaw)
                        print(f"接收到绝对位姿命令: x={x}, y={y}, z={z}, pitch={pitch}, roll={roll}, yaw={yaw}")
                        
                        # 更新缓存
                        self.current_x = x
                        self.current_y = y
                        self.current_z = z
                        self.current_pitch = pitch
                        self.current_roll = roll
                        self.current_yaw = yaw

                        # 构建新姿态（绝对）
                        new_position = airsim.Vector3r(x, y, z)
                        new_orientation = airsim.to_quaternion(pitch, roll, yaw)  # Yaw-Pitch-Roll顺序
                        
                    elif message['channel'].decode('utf-8') == self.pose_change_topic:
                        # 位姿变化命令
                        dx = data_dict.get('dx', 0)
                        dy = data_dict.get('dy', 0)
                        dz = data_dict.get('dz', 0)
                        dpitch = data_dict.get('dpitch', 0)
                        droll = data_dict.get('droll', 0)
                        dyaw = data_dict.get('dyaw', 0)
                        
                        print(f"接收到位姿变化命令: dx={dx}, dy={dy}, dz={dz}, dpitch={dpitch}, droll={droll}, dyaw={dyaw}")
                        
                        # Step 1: 计算新位置（相对偏移从body frame到world frame）
                        # 使用四元数手动旋转向量
                        # 公式：v' = q * v * q^(-1)，但我们可以使用简化的旋转矩阵方法
                        w = current_orientation.w_val
                        x = current_orientation.x_val
                        y = current_orientation.y_val
                        z = current_orientation.z_val
                        
                        # 使用四元数构建旋转矩阵来旋转向量
                        # 这是四元数旋转向量的标准公式
                        rotation_matrix = [
                            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
                        ]
                        
                        # 应用旋转矩阵
                        world_delta_x = rotation_matrix[0][0]*dx + rotation_matrix[0][1]*dy + rotation_matrix[0][2]*dz
                        world_delta_y = rotation_matrix[1][0]*dx + rotation_matrix[1][1]*dy + rotation_matrix[1][2]*dz
                        world_delta_z = rotation_matrix[2][0]*dx + rotation_matrix[2][1]*dy + rotation_matrix[2][2]*dz
                        
                        new_position = airsim.Vector3r(
                            current_position.x_val + world_delta_x,
                            current_position.y_val + world_delta_y,
                            current_position.z_val + world_delta_z
                        )
                        
                        # Step 2: 计算新姿态（相对旋转，使用四元数乘法）
                        # 先创建相对四元数（遵循Yaw-Pitch-Roll顺序）
                        relative_quat = airsim.to_quaternion(dpitch, droll, dyaw)
                        
                        # 四元数乘法：q1 * q2
                        # 公式: (w1, x1, y1, z1) * (w2, x2, y2, z2) = 
                        # (w1*w2 - x1*x2 - y1*y2 - z1*z2,
                        #  w1*x2 + x1*w2 + y1*z2 - z1*y2,
                        #  w1*y2 - x1*z2 + y1*w2 + z1*x2,
                        #  w1*z2 + x1*y2 - y1*x2 + z1*w2)
                        q1 = current_orientation
                        q2 = relative_quat
                        
                        new_w = q1.w_val*q2.w_val - q1.x_val*q2.x_val - q1.y_val*q2.y_val - q1.z_val*q2.z_val
                        new_x = q1.w_val*q2.x_val + q1.x_val*q2.w_val + q1.y_val*q2.z_val - q1.z_val*q2.y_val
                        new_y = q1.w_val*q2.y_val - q1.x_val*q2.z_val + q1.y_val*q2.w_val + q1.z_val*q2.x_val
                        new_z = q1.w_val*q2.z_val + q1.x_val*q2.y_val - q1.y_val*q2.x_val + q1.z_val*q2.w_val
                        
                        new_orientation = airsim.Quaternionr(new_x, new_y, new_z, new_w)
                        
                        # 更新缓存（从新位姿转换回欧拉角）
                        new_euler = airsim.to_eularian_angles(new_orientation)
                        self.current_pitch, self.current_roll, self.current_yaw = new_euler
                        self.current_pitch = self.normalize_angle(self.current_pitch)
                        self.current_roll = self.normalize_angle(self.current_roll)
                        self.current_yaw = self.normalize_angle(self.current_yaw)
                        self.current_x = new_position.x_val
                        self.current_y = new_position.y_val
                        self.current_z = new_position.z_val
                        
                        # 使用calculate_camera_pose函数计算正确的相机位姿
                        # 相机安装在服务航天器质心前方1米处
                        camera_world_pos, camera_orientation, distance_to_origin = calculate_camera_pose(
                            self.current_x, self.current_y, self.current_z,
                            self.current_pitch, self.current_roll, self.current_yaw
                        )
                        print(f"更新后的Airsim世界系下相机位姿: x={camera_world_pos.x_val:.3f}, y={camera_world_pos.y_val:.3f}, z={camera_world_pos.z_val:.3f}, "
                              f"pitch={self.current_pitch:.4f}, roll={self.current_roll:.4f}, yaw={self.current_yaw:.4f}")
                        print(f"相机距离目标航天器{distance_to_origin:.2f}m")
                    
                    # 设置服务航天器新位姿
                    # 先Yaw（偏航，右偏为正），再Pitch（俯仰，抬头为正），最后Roll（滚转，右滚为正）
                    # 使用弧度制，取值范围为-pi到pi
                    new_pose = airsim.Pose(new_position, new_orientation)
                    self.client.simSetVehiclePose(new_pose, True)            
                    
                    time.sleep(0.5)  # 暂停0.5秒，确保服务航天器位姿设置到位了
                    
                    # 生成时间戳（以纳秒为单位）
                    timestamp = str(int(time.time() * 1e9))
                    
                    # 在当前位置收集所有数据
                    self.collect_data(timestamp)
                    
                except json.JSONDecodeError:
                    print("接收到无效的JSON数据")
                except Exception as e:
                    print(f"处理位姿数据时发生错误: {e}")


if __name__ == "__main__":
    collector = AirSimDataCollector()
    collector.fly_control()