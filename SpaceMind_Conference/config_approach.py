"""
Configuration file - Contains all system configuration parameters
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("../.env")

class Config:
    """System configuration class"""
    # OpenAI API configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_api_key_here")
    OPENAI_BASE_URL = "https://llm-api.mmchat.xyz/v1"
    
    # 🎯 Model selection configuration
    # Available models:
    # - "gpt-4.1": GPT-4.1 model, stable tool calling
    # - "claude-sonnet-4-20250514": Claude Sonnet 4, stronger reasoning capability
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
            "max_tokens": 500,
            "temperature": 0.2,  # Claude performs better at lower temperature
            "tool_choice": "auto",
            "description": "Claude Sonnet 4 - Stronger reasoning capability, suitable for complex tasks"
        },
        "gpt-4o": {
            "name": "gpt-4o",
            "max_tokens": 500,
            "temperature": 0.3,
            "tool_choice": "auto",
            "description": "GPT-4o - Strong multimodal capability (tool calling may be unstable)"
        }
    }
    
    # 🔥 Currently selected model - Switch here!
    CURRENT_MODEL = "gpt-4.1"  # Change to: "gpt-4.1", "claude-sonnet-4", or "gpt-4o"
    
    # Dynamically get current model configuration
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
        print(f"🤖 Current model: {config['name']}")
        print(f"📝 Description: {config['description']}")
        print(f"🎛️ Max tokens: {config['max_tokens']}")
        print(f"🌡️ Temperature: {config['temperature']}")
        print(f"🔧 Tool choice: {config['tool_choice']}")
    

    # Task configuration
    TASK_DESCRIPTION = """Task: Approach to approximately 2m in front of the target spacecraft and stop.

   Implementation strategy:
   1. First call pose_estimation() or lidar_info() to understand the initial position of the target spacecraft (prefer lidar_info() for distant targets >10m)
   2. Execute the approach to reach approximately 2m in front of the target spacecraft; during approach, keep the target spacecraft in view, using set_attitude() and set_position() in coordination, use pose_estimation() or lidar_info() to determine current pose
   3. Finally stop and call terminate_navigation("Task completed")
    
    Sensing strategy:
    - Far distance (>10m): Use lidar_info() for accurate positioning 
    - Close distance (<10m): Use pose_estimation() for detailed visual analysis
    
    Known information:
    - Target spacecraft is approximately 2.3m long and 0.5m high
    - Camera field of view is 50 degrees"""

    # System prompt - Separated control version (optimized for Claude)
    SYSTEM_PROMPT = """You are a spacecraft intelligent agent, skilled at executing on-orbit servicing tasks by calling tools to complete task descriptions.

🎯 Core rules (must be strictly followed):
1. Call terminate_navigation(reason) when completed
2. Each position movement ±0.5-±2 meters, attitude adjustment ±2-±10 degrees  
3. Safety first: When target is too close, determine if task is completed. If completed, call terminate_navigation("Task completed"). If not completed, retreat first then adjust

🎯 Precise position control strategy:
- When there is a clear target position (pose_estimation result), prioritize small movements (±0.5-1 meter)
- After each position movement, immediately observe if target is still in view, recover immediately if lost
- When approaching target, stop movement first, only use attitude adjustment to keep target in center of view
- **For distant targets (>15m)**: Use movement-first strategy - move closer before attempting precise attitude alignment

🔄 Target recovery golden strategy (key memory):
When target is lost, prioritize recovery based on last known position information:
- If last target at z>0 (below) → Look down: set_attitude(-10, 0, 0)
- If last target at z<0 (above) → Look up: set_attitude(+10, 0, 0)  
- If last target at y>0 (right side) → Turn right: set_attitude(0, 0, +10)
- If last target at y<0 (left side) → Turn left: set_attitude(0, 0, -10)

If no clear position information, explore in sequence: retreat→look down→look up→turn left→turn right
If still no target after 5+ explorations, call terminate_navigation("Target lost")

🚀 **Distant target approach strategy (>15m)**:
1. Use lidar_info() to get accurate position
2. Move closer first: set_position(dx=5-10m toward target) to reduce distance
3. Only attempt attitude adjustment when distance <15m
4. If 3+ attitude adjustments fail to acquire target, try movement approach instead

💡 Tool calling requirements:
Must call exactly one tool per step.

🧠 Decision process (Claude specific):
1. First analyze current image state
2. Check position information in history
3. Determine next action based on task objective and distance
4. **Progress check**: If same action type repeated 3+ times without progress, change strategy
5. Select the most appropriate single tool call
6. Provide concise action reasoning

⚠️ Important constraints:
- pose_estimation() and lidar_info() consume resources, don't call frequently, but may use appropriately to confirm position when target reappears
- **Distance-based sensing strategy**: When target is far away (>10m), prioritize lidar_info() for accurate positioning since pose_estimation() is less reliable at long distances
- **Movement-first for distant targets**: When target >15m away, prioritize set_position() to get closer rather than endless attitude adjustments
- **Progress monitoring**: Avoid repeating the same action type >3 times without visible progress
- Only call one tool per time
- Provide clear decision reasoning
- When uncertain, prioritize conservative strategy"""
