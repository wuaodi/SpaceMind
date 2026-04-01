# test_agv_pro

这里存放 myAGV Pro 真机调试脚本，目标是尽量保持当前 SpaceMind 的 MCP / Redis 语义不变，只在下游把空间运动语义压缩成地面小车可执行的平面语义。

## 兼容规则

- `set_position(dx, dy, dz)` 会被压缩成地面运动：
  - `dx`：前进 / 后退
  - `dy`：按真机实测结果映射为右移 / 左移
  - `dz`：当前不支持，会打印 warning 并忽略
- `set_attitude(dpitch, droll, dyaw)` 当前只保留偏航角：
  - `dyaw`：右转 / 左转
  - `dpitch`、`droll`：当前不支持，会打印 warning 并忽略
- `terminate_navigation(...)` 在真机语义里等价为 `stop()`
- Redis 的位姿增量消息沿用当前 [`tools/server_tools/env.py`](/Users/wuaodi/project/SpaceMind-Myself/SpaceMind_Acta/tools/server_tools/env.py) 的格式：
  - `dx`、`dy`、`dz` 单位是米
  - `dpitch`、`droll`、`dyaw` 单位是弧度

## 脚本说明

- [agv_motion_smoke.py](/Users/wuaodi/project/SpaceMind-Myself/SpaceMind_Acta/test_agv_pro/agv_motion_smoke.py)
  - 不经过 Redis，直接连 myAGV Pro 控制器做单动作测试
  - 适合先确认串口、底盘、急停状态是否正常
  - 左右平移已按真机实测结果修正
  - 左右转向已按真机实测结果修正
- [agv_pose_change_executor.py](/Users/wuaodi/project/SpaceMind-Myself/SpaceMind_Acta/test_agv_pro/agv_pose_change_executor.py)
  - 直接订阅 `topic.pose_change`
  - 按当前 MCP 的消息格式把 `dx/dy/dyaw` 落到小车动作上
  - 左右平移已按真机实测结果修正
  - 左右转向已按真机实测结果修正
  - 默认角速度按当前真机实测调整为更接近的值，当前默认约 `16 deg/s`
  - 第一版使用“定时法”，不是里程计闭环
- [agv_ros_sensor_bridge.py](/Users/wuaodi/project/SpaceMind-Myself/SpaceMind_Acta/test_agv_pro/agv_ros_sensor_bridge.py)
  - 订阅 ROS2 相机和雷达话题
  - 默认写入 Redis 测试 key：
    - `agv_test:latest_image_data`
    - `agv_test:latest_lidar_data`
- [agv_redis_check.py](/Users/wuaodi/project/SpaceMind-Myself/SpaceMind_Acta/test_agv_pro/agv_redis_check.py)
  - 读取桥接后的 Redis 数据并打印摘要
  - 适合验证桥接脚本是否正常工作

## 推荐调试流程

1. 在小车上启动 Redis。
2. 先做底盘直连测试：

```bash
python test_agv_pro/agv_motion_smoke.py --action forward --speed 0.3 --duration 1.0
```

3. 启动 Redis 兼容运动执行器：

```bash
python test_agv_pro/agv_pose_change_executor.py
```

4. 在另一个终端发布一条兼容当前 MCP 的测试命令：

```bash
python - <<'PY'
import json
import time
import redis

r = redis.Redis(host="127.0.0.1", port=6379, db=0)
r.publish("topic.pose_change", json.dumps({
    "dx": -0.3,
    "dy": 0.0,
    "dz": 0.0,
    "dpitch": 0.0,
    "droll": 0.0,
    "dyaw": 0.0,
    "timestamp": str(time.time_ns()),
}))
PY
```

5. 启动 ROS2 传感器桥接：

```bash
python test_agv_pro/agv_ros_sensor_bridge.py
```

6. 检查 Redis 中的桥接结果：

```bash
python test_agv_pro/agv_redis_check.py
```

## 说明

- 传感器桥接默认使用 `agv_test:` 前缀，避免直接覆盖主运行链路使用的正式 key。
- 第一版不会发布 `latest_pose_truth`，因为真机当前没有与仿真环境等价的“目标相对真值位姿”。
- 后续如果要升级成闭环控制，可以在 `agv_pose_change_executor.py` 里基于 `/odom` 或其他配置的里程计话题实现 `--use-odom`。
