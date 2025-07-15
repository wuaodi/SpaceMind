import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional
import logging
from logging.handlers import RotatingFileHandler

import nest_asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
import os
import redis
import ast
import time
from datetime import datetime

# Import simple memory manager
from simple_memory import SimpleHistoryManager
# Import configuration class
from config_approach_vlm import Config
SERVER_SCRIPT_PATH = "server_en_vlm.py"


# Apply nest_asyncio to allow nested event loops (needed for Jupyter/IPython)
nest_asyncio.apply()

# Load environment variables
load_dotenv("../.env")

# Configure logging system
def setup_logging():
    """Setup logging system with separate log file for each run"""
    # Create log directory if it doesn't exist
    log_dir = "log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"📁 Created log directory: {log_dir}")
    
    # Generate log filename with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"spacemind_{timestamp}.txt")
    
    # Create logger
    logger = logging.getLogger('spacemind_navigation')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers to avoid duplication
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler - no rotation needed since each run gets its own file
    file_handler = logging.FileHandler(
        log_filename, 
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log the log file location
    print(f"📝 Experiment log file: {log_filename}")
    
    return logger

# Initialize logging
logger = setup_logging()


class MCPOpenAIClient:
    """Client for interacting with OpenAI models using MCP tools."""

    def __init__(self, model: str = None, 
                 api_key: Optional[str] = None, 
                 base_url: Optional[str] = None):
        """Initialize the OpenAI MCP client.

        Args:
            model: The OpenAI model to use.
            api_key: The API key to use. If not provided, will try to get from METACHAT_API_KEY env var.
            base_url: The base URL for the API. If not provided, will use default.
        """
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        # Configure OpenAI client with custom API key and base URL
        if api_key is None:
            api_key = Config.OPENAI_API_KEY
        
        if base_url is None:
            base_url = Config.OPENAI_BASE_URL
            
        self.openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = Config.OPENAI_MODEL
        self.stdio: Optional[Any] = None
        self.write: Optional[Any] = None
        
        # Redis configuration
        self.redis_client = None
        self.image_topic = 'topic.img'
        
        # Initialize simple history
        self.history = SimpleHistoryManager()

    async def connect_to_server(self, server_script_path: str = SERVER_SCRIPT_PATH):
        """Connect to an MCP server.

        Args:
            server_script_path: Path to the server script.
        """
        # Server configuration
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
        )

        # Connect to the server
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        # Initialize the connection
        await self.session.initialize()

        # List available tools
        tools_result = await self.session.list_tools()
        logger.info("Connected to server with tools:")
        for tool in tools_result.tools:
            logger.info(f"{tool.name}")

    def setup_redis_connection(self):
        """Setup Redis connection"""
        try:
            self.redis_client = redis.Redis(host='127.0.0.1', port=6379)
            self.redis_client.ping()
            logger.info("✅ Successfully connected to Redis server")
            return True
        except redis.ConnectionError:
            logger.error("❌ Cannot connect to Redis server")
            return False
        except Exception as e:
            logger.error(f"❌ Redis connection error: {e}")
            return False

    def wait_for_image(self):
        """Wait for receiving an image (synchronous mode)"""
        if not self.redis_client:
            if not self.setup_redis_connection():
                return None
                
        logger.info("⏳ Waiting for image data...")
        
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(self.image_topic)
        
        try:
            start_time = time.time()
            
            for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        # Parse received image data
                        message_data = message['data'].decode('utf-8')
                        message_dict = ast.literal_eval(message_data)
                        
                        image_data = {
                            'name': message_dict['name'],
                            'timestamp': message_dict['timestamp'],
                            'width': message_dict['width'],
                            'height': message_dict['height'],
                            'data': message_dict['data']  # base64 encoded image data
                        }
                        
                        logger.info(f"📸 Received image: {message_dict['name']} ({message_dict['width']}x{message_dict['height']})")
                        return image_data
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing image data: {e}")
                        continue
                
                    
        except Exception as e:
            logger.error(f"❌ Error during image reception: {e}")
        finally:
            pubsub.unsubscribe(self.image_topic)
            pubsub.close()
            
        return None

    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from the MCP server in OpenAI format.

        Returns:
            A list of tools in OpenAI format.
        """
        tools_result = await self.session.list_tools()
        # Return tools in OpenAI specified format
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools_result.tools
        ]

    async def analyze_image_with_tools(self, image_data: dict, task: str = "Please analyze the image and decide the next action", system_prompt: str = "Please analyze the image and decide the next action") -> dict:
        """Analyze image and use tools to decide next action
        
        Args:
            image_data: Image data dictionary containing name, data, etc.
            task: Task description
            
        Returns:
            Dictionary containing analysis results and actions
        """
        if not image_data:
            return {"error": "No available image data"}
        
        # Construct image format for GPT-4.1 API
        image_url = f"data:image/png;base64,{image_data['data']}"
        
        # Get MCP tools
        tools = await self.get_mcp_tools()
        
        # Get history context
        history_context = self.history.get_recent_context()
        
        # Enhanced system prompt with history information
        enhanced_system_prompt = f"""{system_prompt}
{history_context}"""
        
        try:
            logger.info(f"🔍 Analyzing image and deciding: {image_data['name']}")
            logger.info(f"📚 History: Total {self.history.step_count} steps")
            
            # Call GPT-4o Vision API with tools
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": enhanced_system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": task
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                tools=tools,
                tool_choice=Config.MODEL_TOOL_CHOICE,
                max_tokens=Config.MODEL_MAX_TOKENS,
                temperature=Config.MODEL_TEMPERATURE
            )
            
            assistant_message = response.choices[0].message
            
            result = {
                "analysis": assistant_message.content,
                "tool_called": False,
                "tool_name": None,
                "pose_change": None,
                "terminate_navigation": False,
                "terminate_reason": None
            }
            
            # Handle tool calls
            if assistant_message.tool_calls:
                tool_results = {}  # Store all tool results
                segmentation_result = None  # Store segmentation results
                
                for tool_call in assistant_message.tool_calls:
                    logger.info(f"🔧 Calling tool: {tool_call.function.name}")
                    arguments = json.loads(tool_call.function.arguments)
                    
                    # Execute tool call
                    tool_result = await self.session.call_tool(
                        tool_call.function.name,
                        arguments=arguments,
                    )
                    
                    # Extract CallToolResult content (simplified processing)
                    if hasattr(tool_result, 'content'):
                        # Handle content list
                        if isinstance(tool_result.content, list) and len(tool_result.content) > 0:
                            # Get text from first content item
                            actual_result = tool_result.content[0].text if hasattr(tool_result.content[0], 'text') else str(tool_result.content[0])
                        else:
                            actual_result = str(tool_result.content)
                    else:
                        actual_result = str(tool_result)
                    
                    # Save tool results
                    tool_results[tool_call.function.name] = actual_result
                    
                    result["tool_called"] = True
                    result["tool_name"] = tool_call.function.name
                    result["tool_result"] = actual_result  # Save extracted result
                    
                    if tool_call.function.name == "set_position":
                        result["position_change"] = json.loads(tool_call.function.arguments)
                        logger.info(f"✅ Position set: {result['position_change']}")
                    elif tool_call.function.name == "set_attitude":
                        result["attitude_change"] = json.loads(tool_call.function.arguments)
                        logger.info(f"✅ Attitude set: {result['attitude_change']}")
                    elif tool_call.function.name == "set_pose_change":  # Keep compatibility
                        result["pose_change"] = json.loads(tool_call.function.arguments)
                        logger.info(f"✅ Pose set: {result['pose_change']}")
                    elif tool_call.function.name == "terminate_navigation":
                        result["terminate_navigation"] = True
                        args = json.loads(tool_call.function.arguments)
                        result["terminate_reason"] = args.get("reason", "Task completed")
                    elif tool_call.function.name == "pose_estimation":
                        # Record pose estimation results
                        if isinstance(actual_result, str) and "Distance:" in actual_result:
                            # Extract distance information from string
                            import re
                            distance_match = re.search(r'Distance:\s*(\d+\.?\d*)m', actual_result)
                            if distance_match:
                                distance = distance_match.group(1)
                                logger.info(f"📏 Target distance: {distance}m")
                    elif tool_call.function.name == "lidar_info":
                        # Record LiDAR results
                        if isinstance(actual_result, str) and "Point count:" in actual_result:
                            # Extract point count from string
                            import re
                            count_match = re.search(r'Point count:\s*(\d+) points', actual_result)
                            if count_match:
                                count = count_match.group(1)
                                logger.info(f"📡 LiDAR point cloud: {count} points")
                    elif tool_call.function.name == "image_bright":
                        # Record image brightness results
                        if isinstance(actual_result, str) and "Foreground average brightness:" in actual_result:
                            # Extract brightness information from string
                            import re
                            brightness_match = re.search(r'Foreground average brightness:\s*(\d+\.?\d*)', actual_result)
                            if brightness_match:
                                brightness = brightness_match.group(1)
                                logger.info(f"💡 Image brightness: {brightness}")
                    elif tool_call.function.name == "part_segmentation":
                        # Save segmentation results for subsequent analysis
                        if isinstance(actual_result, dict) and 'segmentation_image' in actual_result:
                            segmentation_result = actual_result
                            logger.info(f"🎯 Part segmentation completed, ready to analyze segmentation results")
                
                # If part_segmentation tool was called, let LVM analyze segmentation results
                if segmentation_result:
                    logger.info("🎯 Analyzing segmentation image for better decision making...")
                    
                    # Construct segmentation image URL
                    seg_image_url = f"data:image/png;base64,{segmentation_result['segmentation_image']}"
                    
                    # Update history context
                    updated_history = self.history.get_recent_context()
                    
                    # Construct prompt for analyzing segmentation results
                    segmentation_analysis_prompt = f"""Based on part segmentation results, please analyze: 1. Identified spacecraft components (such as solar panels, main body, antennas, etc.) 2. Relative position and orientation of each component 3. Based on mission objectives, suggest next movement strategy
                    Mission objective: {task}
                    History context: {updated_history}
                    Please decide the next action (move or terminate) based on segmentation image analysis."""
                    
                    # Call LVM to analyze segmentation results
                    seg_response = await self.openai_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": enhanced_system_prompt
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": segmentation_analysis_prompt
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": image_url,
                                            "detail": "high"
                                        }
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": seg_image_url,
                                            "detail": "high"
                                        }
                                    }
                                ]
                            }
                        ],
                        tools=tools,
                        tool_choice=Config.MODEL_TOOL_CHOICE,
                        max_tokens=Config.MODEL_MAX_TOKENS,
                        temperature=Config.MODEL_TEMPERATURE
                    )
                    
                    seg_assistant_message = seg_response.choices[0].message
                    
                    # Update analysis results with segmentation analysis
                    result["analysis"] = f"{assistant_message.content}\n\nSegmentation Image Analysis:\n{seg_assistant_message.content}"
                    
                    # Handle tool calls based on segmentation analysis
                    if seg_assistant_message.tool_calls:
                        for tool_call in seg_assistant_message.tool_calls:
                            logger.info(f"🔧 Calling tool based on segmentation analysis: {tool_call.function.name}")
                            
                            # Execute tool call
                            tool_result = await self.session.call_tool(
                                tool_call.function.name,
                                arguments=json.loads(tool_call.function.arguments),
                            )
                            
                            if tool_call.function.name == "set_pose_change":
                                result["pose_change"] = json.loads(tool_call.function.arguments)
                                logger.info(f"✅ Pose set based on segmentation analysis: {result['pose_change']}")
                            elif tool_call.function.name == "set_position":
                                result["position_change"] = json.loads(tool_call.function.arguments)
                                logger.info(f"✅ Position set based on segmentation analysis: {result['position_change']}")
                            elif tool_call.function.name == "set_attitude":
                                result["attitude_change"] = json.loads(tool_call.function.arguments)
                                logger.info(f"✅ Attitude set based on segmentation analysis: {result['attitude_change']}")
                            elif tool_call.function.name == "terminate_navigation":
                                result["terminate_navigation"] = True
                                args = json.loads(tool_call.function.arguments)
                                result["terminate_reason"] = args.get("reason", "Task completed")
                
                # If multiple tool calls, save all results
                if len(tool_results) > 1:
                    result["all_tool_results"] = tool_results
            
            # Update history
            tool_arguments = None
            if result["tool_called"]:
                if result["tool_name"] == "set_pose_change":
                    tool_arguments = result.get("pose_change")
                elif result["tool_name"] == "set_position":
                    tool_arguments = result.get("position_change")
                elif result["tool_name"] == "set_attitude":
                    tool_arguments = result.get("attitude_change")
                elif assistant_message.tool_calls:
                    # Get parameters of first tool call
                    tool_arguments = json.loads(assistant_message.tool_calls[0].function.arguments)
            
            self.history.add_step(
                image_name=image_data['name'],
                analysis_result=assistant_message.content,
                tool_called=result["tool_called"],
                tool_name=result["tool_name"],
                tool_arguments=tool_arguments,
                tool_result=result.get("tool_result") if result["tool_called"] else None
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error during analysis and decision: {e}")
            return {"error": str(e)}


    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()
        if self.redis_client:
            self.redis_client.close()


async def autonomous_navigation_demo():
    """Autonomous navigation demo: receive image → analyze → decide → move → loop"""
    logger.info("🚀 Starting autonomous navigation demo")
    
    # Show current model configuration
    logger.info("\n" + "="*50)
    logger.info("🤖 Model Configuration")
    logger.info("="*50)
    Config.show_current_model_info()
    logger.info("="*50 + "\n")
    
    client = MCPOpenAIClient()
    
    # Connect to MCP server
    await client.connect_to_server(SERVER_SCRIPT_PATH)
    
    # Setup Redis connection
    if not client.setup_redis_connection():
        logger.error("❌ Cannot connect to Redis")
        return
    
    # Use system prompt and task description from config file
    system_prompt = Config.SYSTEM_PROMPT
    task = Config.TASK_DESCRIPTION
    
    logger.info(f"📋 Mission objective: {task}")
    logger.info("🔄 Starting autonomous navigation loop...")
    
    try:
        step = 0
        no_tool_count = 0  # Count consecutive steps without tool calls
        while True:
            step += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"📍 Step {step}")
            
            # 1. Wait for image
            image_data = client.wait_for_image()
            
            # 2. Analyze image and decide
            logger.info("🤖 Analyzing image and deciding next action...")
            
            result = await client.analyze_image_with_tools(image_data, task, system_prompt)
              
            # 3. Show analysis results
            logger.info(f"📝 Analysis: {result['analysis']}")
            if result["tool_called"]:
                logger.info(f"🔧 Tool called: {result['tool_name']}")
            else:
                logger.info("🚫 No tool called")
            
            # 4. Handle spacecraft operation related tool call results
            if result["tool_called"]:
                no_tool_count = 0  # Reset counter
                if result["terminate_navigation"]:
                    # Task completed, terminate navigation
                    logger.info(f"\n🏁 Task completed!")
                    logger.info(f"📝 Termination reason: {result['terminate_reason']}")
                    
                    # Show movement history summary
                    move_summary = client.history.get_move_summary()
                    logger.info(f"\n📊 {move_summary}")
                    break
                elif result.get("pose_change"):
                    # Move to new position (backward compatibility)
                    logger.info(f"\n🎯 Decided to move to new position: {result['pose_change']}")
                    logger.info("⏳ Waiting for service spacecraft to move to new position...")
                elif result.get("position_change"):
                    # Position change
                    logger.info(f"\n🎯 Decided to change position: {result['position_change']}")
                    logger.info("⏳ Waiting for service spacecraft to move to new position...")
                elif result.get("attitude_change"):
                    # Attitude change
                    logger.info(f"\n🎯 Decided to adjust attitude: {result['attitude_change']}")
                    logger.info("⏳ Waiting for service spacecraft to adjust attitude...")
                else:
                    # Called other analysis tools (like pose_estimation, lidar_info, image_bright, etc.)
                    logger.info(f"\n📊 Called analysis tool: {result['tool_name']}")
                    # For analysis tools, need to call set_position tool to maintain position for new image
                    await client.session.call_tool("set_position", arguments={"dx": 0, "dy": 0, "dz": 0})
                    # LVM will decide next step based on analysis results
                    logger.info("🔄 Waiting for LVM to decide based on analysis results...")
            else:
                no_tool_count += 1  # No tool called, increment counter
                if no_tool_count >= 5:  # If 5 consecutive steps without tool calls
                    logger.warning("\n⚠️ 5 consecutive steps without tool calls, possible issue")
                    logger.warning("Check: 1) API status 2) Prompt clarity 3) Tool definitions")
                    break
                logger.info("\n💤 Decided to maintain current position, continue observing")
                # Call set_position tool to maintain position for new image
                await client.session.call_tool("set_position", arguments={"dx": 0, "dy": 0, "dz": 0})
            
    except KeyboardInterrupt:
        logger.warning("\n🛑 Navigation interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error during navigation: {e}")
    # When try block completes normally, finally block will execute
    finally:
        await client.cleanup()
        logger.info("👋 Navigation demo ended")


if __name__ == "__main__":
    # Select run mode
    logger.info("Autonomous navigation demo")
    asyncio.run(autonomous_navigation_demo())
