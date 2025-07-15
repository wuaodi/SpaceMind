# SpaceMind

SpaceMind is an intelligent spacecraft autonomous navigation system that enables spacecraft to perform complex on-orbit servicing tasks through vision-based decision making and Large Vision Model (LVM) integration.

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

### Mission Types

#### Perception Mission
Analyzes target spacecraft to determine type, status, and function.
- Configuration: `config_perception.py`
- Focus: Detailed spacecraft analysis and classification

#### Approach Mission  
Navigates to a specific distance from target spacecraft.
- Configuration: `config_approach.py` or `config_approach_vlm.py`
- Focus: Precision navigation and positioning

#### Fly-Around Mission
Performs orbital maneuvering around target spacecraft.
- Configuration: `config_around.py`
- Focus: Complex trajectory planning and execution

## Model Configuration

The system supports multiple AI models:

- **GPT-4.1**: Stable tool calling, fast response
- **Claude Sonnet 4**: Superior reasoning for complex tasks
- **GPT-4o**: Strong multimodal capabilities

Select model by modifying `CURRENT_MODEL` in configuration files.

## Coordinate System

The system uses spacecraft body coordinate system:
- **X-axis**: Forward (+) / Backward (-)
- **Y-axis**: Right (+) / Left (-)  
- **Z-axis**: Down (+) / Up (-)

## Safety Features

- Conservative movement strategies
- Target loss recovery protocols
- Distance-based safety constraints
- Mission termination on completion or failure

## Development

### Adding New Tools
1. Define tool functions in MCP server files
2. Add appropriate tool documentation and examples
3. Update configuration prompts as needed

### Customizing Missions
1. Create new configuration file based on existing templates
2. Define task-specific prompts and parameters
3. Update import in `host.py`

## License

This project is provided as-is for research and educational purposes.

## Contributing

Please ensure all API keys and personal information are removed before contributing. 