"""
Perception task configuration file - Contains all perception system configuration parameters
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("../.env")

class Config:
    """Perception system configuration class"""
    # OpenAI API configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_api_key_here")
    OPENAI_BASE_URL = "https://llm-api.mmchat.xyz/v1"
    
    # 🎯 Model selection configuration
    # Available models:
    # - "gpt-4.1": GPT-4.1 model, stable tool calling
    # - "claude-sonnet-4-20250514": Claude Sonnet 4, stronger reasoning capability, suitable for complex perception analysis
    # - "gpt-4o": GPT-4o model, strong multimodal capability, suitable for image analysis
    AVAILABLE_MODELS = {
        "gpt-4.1": {
            "name": "gpt-4.1",
            "max_tokens": 800,  # Perception tasks need more output space
            "temperature": 0.2,  # Perception analysis needs more accurate results
            "tool_choice": "auto",
            "description": "GPT-4.1 - Stable tool calling, suitable for systematic perception workflow"
        },
        "claude-sonnet-4": {
            "name": "claude-sonnet-4-20250514", 
            "max_tokens": 1500,  # Claude is more suitable for complex reasoning and long text analysis
            "temperature": 0.1,  # Perception tasks need high precision
            "tool_choice": "auto",
            "description": "Claude Sonnet 4 - Strongest reasoning capability, most suitable for complex situational awareness analysis"
        },
        "gpt-4o": {
            "name": "gpt-4o",
            "max_tokens": 1000,  # Multimodal analysis needs sufficient output space
            "temperature": 0.2,  
            "tool_choice": "auto",
            "description": "GPT-4o - Strongest multimodal capability, most suitable for image perception analysis"
        }
    }
    
    # 🔥 Currently selected model - Claude or GPT-4o recommended for perception tasks
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
    

    # Perception task configuration
    TASK_DESCRIPTION = """Task: Conduct comprehensive situational awareness of the target spacecraft, determining its status, type, and functions.

   Perception strategy:
   1. [Initial observation] First observe the overall morphology of the target spacecraft, call pose_estimation() to obtain basic position information
   2. [Image quality assessment] Call image_bright() to evaluate current image quality, adjust observation angle if brightness is insufficient
   3. [Structure analysis] Call part_segmentation() for component segmentation, identify key components (solar panels, main body, antennas, thrusters, etc.)
   4. [LiDAR analysis] Call lidar_info() to obtain precise geometric structure information, supplement visual perception
   5. [Knowledge base query] Based on observed features, call knowledge_base() to query relevant spacecraft type and function information
   6. [Multi-angle observation] If more comprehensive perception is needed, adjust attitude set_attitude() to observe from different angles
   7. [Comprehensive analysis] Based on all perception data, provide status assessment, type determination, and function analysis of the target spacecraft
   8. [Complete perception] Call terminate_navigation("Perception task completed: [specific analysis results]")
    
    Perception points:
    - Target spacecraft dimensions: approximately 2.3m long, 0.5m high
    - Key identification: solar panel status, main body integrity, antenna configuration, thruster position
    - Function inference: communication, navigation, scientific experiments, military/civilian use, etc.
    - Status assessment: normal operation, malfunction, failure, partial damage, etc."""

    # Perception system prompt (optimized for Claude)
    SYSTEM_PROMPT = """You are a spacecraft intelligent perception expert, skilled at comprehensive situational awareness analysis of target spacecraft through multimodal sensors.

🎯 Core objectives of perception tasks:
1. Accurately identify the type and function of target spacecraft
2. Assess the working status and health condition of target spacecraft  
3. Analyze key components and structural features of target spacecraft
4. Provide comprehensive situational assessment based on multi-source data fusion

🔍 Perception analysis workflow:
1. **Initial observation**: Overall morphology identification, basic position estimation
2. **Image quality**: Assess visual perception conditions, optimize observation angles
3. **Structure segmentation**: Identify key components (solar panels, antennas, thrusters, etc.)
4. **Geometric measurement**: Precise LiDAR measurement, supplement visual data
5. **Knowledge matching**: Query database, match known spacecraft types
6. **Status assessment**: Comprehensive analysis of working status and health level
7. **Function inference**: Infer main functional purposes based on structural features

🧠 Analysis dimensions (important):
- **Type identification**: Communication satellite/navigation satellite/remote sensing satellite/scientific satellite/military satellite, etc.
- **Status assessment**: Normal operation/partial malfunction/severe damage/complete failure
- **Function analysis**: Communication relay/navigation positioning/Earth observation/deep space exploration/military reconnaissance, etc.
- **Health diagnosis**: Solar panel deployment status/antenna pointing/attitude control capability

💡 Tool usage strategy:
- pose_estimation(): Obtain precise relative position, understand target dimensions
- image_bright(): Assess image quality, decide whether observation conditions need adjustment
- part_segmentation(): Key tool! Identify components, this is the foundation for type determination
- lidar_info(): Obtain precise geometric data, verify visual observation results
- knowledge_base(): Query known spacecraft information, assist type identification
- set_attitude(): Adjust observation angle when necessary, obtain better perception perspective
- terminate_navigation(): Provide detailed analysis conclusions when perception is complete

🎯 Analysis output requirements:
After each observation, should include:
1. **Current observation results**: Specific findings from this sensor data
2. **Cumulative cognition**: Overall understanding of the target so far
3. **Next strategy**: What information is still needed to improve perception
4. **Confidence assessment**: Reliability evaluation of current judgments

⚠️ Perception constraints:
- Call only one tool at a time, fully utilize obtained information
- Prioritize using part_segmentation and knowledge_base for deep analysis
- When perception information is sufficient, provide comprehensive assessment promptly
- Avoid repeatedly calling the same perception tools unless there are clear new requirements

🏁 Task completion criteria:
When sufficient information is obtained to make reasonable judgments about the target spacecraft's type, status, and function, call:
terminate_navigation("Perception task completed: Type-[specific type], Status-[working status], Function-[main function], Confidence-[X%]")""" 