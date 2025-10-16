# Demo 📹

[![Demo Video](https://img.shields.io/badge/View%20Demo-SpaceMind%20Agent-blue?style=for-the-badge&logo=youtube)](https://sites.google.com/view/spacemind-agent/)

You can find more video demos here: [SpaceMind Agent Demo Site](https://sites.google.com/view/spacemind-agent/)

## Project Overview

This system simulates autonomous spacecraft navigation using AirSim simulation environment, combined with advanced AI models (GPT-4V, Claude Sonnet 4) for real-time decision making and control.

## Key Features

- **Autonomous Navigation**: Vision-based spacecraft navigation and control
- **Multiple Task Modes**: 
  - Perception tasks (spacecraft identification and analysis)
  - Approach maneuvers (precision navigation to target)
  - Fly-around operations (orbital maneuvering)
- **LVM Integration**: Support for multiple AI models (GPT-4.1, Claude Sonnet 4, GPT-4o)
- **Real-time Processing**: Redis-based communication for sensor data streaming
- **AirSim Integration**: Realistic space environment simulation

## System Architecture

### Core Components

1. **Configuration Files**: Model and task-specific configurations
   - `config_perception.py` - Perception task settings
   - `config_approach.py` - Approach maneuver settings
   - `config_approach_vlm.py` - Vision-only approach settings
   - `config_around.py` - Fly-around operation settings

2. **MCP Servers**: Tool providers for spacecraft control
   - `server_en.py` - Standard sensor-assisted navigation tools
   - `server_en_vlm.py` - Vision-only navigation tools

3. **Main Controller**: 
   - `host.py` - Main system controller and AI model interface

4. **Memory Management**:
   - `simple_memory.py` - History tracking and context management

5. **AirSim Integration**:
   - `Airsim-collector/fly_redis.py` - Data collection and spacecraft control interface

## Available Tools

### Navigation Control
- `set_position(dx, dy, dz)` - Spacecraft position control
- `set_attitude(dpitch, droll, dyaw)` - Spacecraft attitude control
- `terminate_navigation(reason)` - Mission completion

### Sensor Systems
- `pose_estimation()` - Visual pose estimation of target spacecraft
- `lidar_info()` - LiDAR distance and geometry measurements
- `part_segmentation()` - Component identification and segmentation
- `knowledge_base()` - Spacecraft database queries

## Installation

1. **Prerequisites**:
   - Python 3.10
   - Redis server
   - AirSim simulation environment

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup**:
   - Create a `.env` file with your API keys:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

4. **Redis Configuration**:
   - Ensure Redis server is running on localhost:6379

## Usage

### Basic Operation

1. **Start AirSim Simulation**:
   - Launch AirSim with spacecraft environment
   - Ensure target and service spacecraft are properly positioned

2. **Start Data Collector**:
   ```bash
   cd Airsim-collector
   python fly_redis.py
   ```

3. **Configure Mission**:
   - Choose appropriate configuration file based on mission type
   - Modify `host.py` to import desired config

4. **Launch Navigation System**:
   ```bash
   python host.py
   ```

## Citation

@inproceedings{wu2025spacemind,
  title={SpaceMind: An MCP-based Agent Architecture Fusing Large and Small Models for On-orbit Servicing},
  author={Wu, Aodi and Han, Haodong and Luo, Xubo and Wan, Xue},
  booktitle={IAA Conference on AI in and for Space},
  year={2025},
  address={Suzhou, China},
  month={November},
  url={https://github.com/[your-username]/SpaceMind}
}
