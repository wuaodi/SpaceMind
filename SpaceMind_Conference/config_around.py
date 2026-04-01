"""
Configuration File - Contains all system configuration parameters (English Version)
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("../.env")

class Config:
    """System Configuration Class"""
    # OpenAI API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_api_key_here")
    OPENAI_BASE_URL = "https://llm-api.mmchat.xyz/v1"
    
    # 🎯 Model Selection Configuration
    # Available models:
    # - "gpt-4.1": GPT-4.1 model, stable tool calling
    # - "claude-sonnet-4-20250514": Claude Sonnet 4, superior reasoning capabilities
    # - "gpt-4o": GPT-4o model (tool calling may be unstable)
    AVAILABLE_MODELS = {
        "gpt-4.1": {
            "name": "gpt-4.1",
            "max_tokens": 500,
            "temperature": 0.3,
            "tool_choice": "auto",
            "description": "GPT-4.1 - Stable tool calling, fast response"
        },
        "claude-sonnet-4": {
            "name": "claude-sonnet-4-20250514", 
            "max_tokens": 1000,  # Claude typically supports longer outputs
            "temperature": 0.2,  # Claude performs better at lower temperatures
            "tool_choice": "auto",
            "description": "Claude Sonnet 4 - Superior reasoning capabilities, ideal for complex tasks"
        },
        "gpt-4o": {
            "name": "gpt-4o",
            "max_tokens": 500,
            "temperature": 0.3,
            "tool_choice": "auto",
            "description": "GPT-4o - Strong multimodal capabilities (tool calling may be unstable)"
        }
    }
    
    # 🔥 Currently selected model - Switch here!
    CURRENT_MODEL = "gpt-4.1"  # Change to: "gpt-4.1", "claude-sonnet-4", or "gpt-4o"
    
    # Dynamic model configuration retrieval
    @classmethod
    def get_model_config(cls):
        return cls.AVAILABLE_MODELS[cls.CURRENT_MODEL]
    
    # Dynamic configuration application
    _current_config = AVAILABLE_MODELS[CURRENT_MODEL]
    
    # Apply current model configuration
    OPENAI_MODEL = _current_config["name"]
    MODEL_MAX_TOKENS = _current_config["max_tokens"] 
    MODEL_TEMPERATURE = _current_config["temperature"]
    MODEL_TOOL_CHOICE = _current_config["tool_choice"]
    
    # Display current model information
    @classmethod
    def show_current_model_info(cls):
        config = cls.AVAILABLE_MODELS[cls.CURRENT_MODEL]
        print(f"🤖 Current Model: {config['name']}")
        print(f"📝 Description: {config['description']}")
        print(f"🎛️ Max Tokens: {config['max_tokens']}")
        print(f"🌡️ Temperature: {config['temperature']}")
        print(f"🔧 Tool Choice: {config['tool_choice']}")
    
    # Mission Configuration
    TASK_DESCRIPTION = """Mission: Fly around to reach the position directly above the target spacecraft, and finally stop at approximately 2m distance from the target spacecraft.

Implementation Strategy:
1. First call pose_estimation() or lidar_info() to understand the initial position of the target spacecraft
2. Execute fly-around maneuver to reach the position directly above the target spacecraft; during fly-around, keep the target spacecraft in the field of view, using set_attitude() and set_position() in coordination, and use pose_estimation() or lidar_info() to determine current pose
3. Finally execute approach maneuver, reaching approximately 2m distance from the target spacecraft and stop

Known Information:
- Target spacecraft is approximately 2.3m long and 0.5m high
- Camera field of view is 50 degrees"""

    # System Prompt - Separated Control Version (Optimized for Claude)
    SYSTEM_PROMPT = """You are a spacecraft intelligent agent, specializing in executing space on-orbit service missions, completing mission descriptions through tool calling.

🎯 Core Rules (Must strictly follow):
1. Call terminate_navigation(reason) when mission is complete
2. Each position movement ±0.5-±2 meters, attitude adjustment ±2-±10 degrees  
3. Safety first: When target is too close, determine if mission is complete. If complete, call terminate_navigation("Mission Complete"). If not complete, retreat first then adjust

🎯 Precision Position Control Strategy:
- When clear target position is available (from pose_estimation results), prioritize small movements (±0.5-1 meters)
- After each position movement, immediately observe if target remains in field of view, recover immediately if lost
- When approaching target, stop movement first, only use attitude adjustment to keep target in view center

🔄 Target Recovery Golden Strategy (Critical Memory):
When target is lost, prioritize recovery based on last known position information:
- If last target at z>0 (below) → Look down: set_attitude(-10, 0, 0)
- If last target at z<0 (above) → Look up: set_attitude(+10, 0, 0)  
- If last target at y>0 (right side) → Turn right: set_attitude(0, 0, +10)
- If last target at y<0 (left side) → Turn left: set_attitude(0, 0, -10)

If no clear position information, explore in sequence: retreat→look down→look up→turn left→turn right
If still no target after 5+ explorations, call terminate_navigation("Target lost")

💡 Tool Calling Requirements:
Must call exactly one tool per step. Available tools:
- set_position(dx, dy, dz): Position movement
- set_attitude(dpitch, droll, dyaw): Attitude adjustment  
- pose_estimation(): Get target position information
- lidar_info(): Get distance information
- terminate_navigation(reason): Complete mission

🧠 Decision Process (Claude-specific):
1. First analyze current image status
2. Review position information in history records
3. Determine next action based on mission objectives
4. Select the most appropriate single tool call
5. Provide concise reasoning for action

⚠️ Important Constraints:
- pose_estimation() and lidar_info() consume resources, don't call frequently, but may use appropriately to confirm position when target reappears
- Only call one tool per step
- Provide clear decision reasoning
- When uncertain, prioritize conservative strategy""" 