import os
import json
from mcp.server.fastmcp import FastMCP

# Pre-load all required modules to avoid delays during first call
import redis
import numpy as np
import ast
import base64
from io import BytesIO
from PIL import Image
import time
import math

# Create an MCP server
mcp = FastMCP(
    name="Knowledge Base",
    host="0.0.0.0",  # only used for SSE transport (localhost)
    port=8050,  # only used for SSE transport (set this to any port)
)

# Multiple tools can be defined in one server
@mcp.tool()
def set_position(dx: float, dy: float, dz: float) -> str:
    """Control spacecraft position change (translation control)
    
    This tool is specifically used to control the position movement of the service spacecraft, 
    corresponding to the actual satellite's thruster control system.
    All changes are increments relative to the current position.
    
    Coordinate system definition (spacecraft body coordinate system):
    - X-axis: Forward direction is positive (+dx), backward is negative (-dx)
    - Y-axis: Right direction is positive (+dy), left is negative (-dy)  
    - Z-axis: Downward direction is positive (+dz), upward is negative (-dz)
    
    🎯 CRITICAL MOVEMENT DECISION EXAMPLES:
    If target is at relative position (x=20.18m, y=0.00m, z=5.29m):
    - Target is 20.18m FORWARD and 5.29m BELOW
    - To approach target: set_position(dx=+2.0, dy=0, dz=+1.0)  # Move forward and down
    - To move away: set_position(dx=-2.0, dy=0, dz=-1.0)  # Move backward and up
    
    Common scenarios:
    - Target ahead and above (x>0, z<0): dx>0, dz<0 (forward, up)
    - Target ahead and below (x>0, z>0): dx>0, dz>0 (forward, down) 
    - Target behind and below (x<0, z>0): dx<0, dz>0 (backward, down)
    - Target on right side (y>0): dy>0 (move right)
    - Target on left side (y<0): dy<0 (move left)
    
    ⚠️ PARAMETER VERIFICATION:
    Before calling, ask yourself:
    - Am I moving in the CORRECT direction toward the target?
    - If target is at x=+20m, should dx be positive? YES!
    - If target is at z=+5m, should dz be positive? YES!
    
    Args:
        dx: X-direction position change, unit: meters, positive for forward, negative for backward
        dy: Y-direction position change, unit: meters, positive for right, negative for left
        dz: Z-direction position change, unit: meters, positive for down, negative for up
    
    Returns:
        Operation result description string
    """
    try:    
        # Connect to Redis
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
        redis_client.ping()
        
        # Publish position change (attitude parameters are 0)
        pose_change = {
            'dx': round(dx, 3),
            'dy': round(dy, 3),
            'dz': round(dz, 3),
            'dpitch': 0.0,
            'droll': 0.0,
            'dyaw': 0.0,
            'timestamp': str(int(time.time() * 1e9))
        }
        
        message = json.dumps(pose_change)
        redis_client.publish('topic.pose_change', message)
        redis_client.close()
        
        return f"Position change completed: dx={dx}m, dy={dy}m, dz={dz}m"
        
    except Exception as e:
        return f"Failed to set position: {str(e)}"

@mcp.tool()
def set_attitude(dpitch: float, droll: float, dyaw: float) -> str:
    """Control spacecraft attitude change (rotation control)
    
    This tool is specifically used to control the attitude adjustment of the service spacecraft, 
    corresponding to the actual satellite's reaction wheel/gyroscope control system.
    All changes are increments relative to the current attitude.
    
    Attitude changes (Euler angles, in degrees):
    - dpitch: Pitch angle change, nose up is positive (+), nose down is negative (-)
    - droll: Roll angle change, right roll is positive (+), left roll is negative (-)
    - dyaw: Yaw angle change, right turn is positive (+), left turn is negative (-)
    
    🔥 Important: Relationship between attitude control and coordinate system (Z-axis downward is positive)
    Target recovery strategy:
    - If target at z < 0 (target above spacecraft) → need to pitch up → dpitch > 0 ✅
    - If target at z > 0 (target below spacecraft) → need to pitch down → dpitch < 0 ✅
    - If target at y < 0 (target on left side) → need to turn left → dyaw < 0 ✅
    - If target at y > 0 (target on right side) → need to turn right → dyaw > 0 ✅
    
    Remember: z=-1.8m means target is above, need to pitch up (dpitch > 0)!
    
    🎯 CONCRETE ATTITUDE CONTROL EXAMPLES:
    If target is at relative position (x=20.18m, y=0.00m, z=5.29m):
    - Target is BELOW (z=+5.29m), so need to look DOWN
    - Action: set_attitude(dpitch=-10, droll=0, dyaw=0)  # Pitch down to look at target
    
    If target is at relative position (x=15.0m, y=-3.0m, z=-2.0m):
    - Target is ABOVE (z=-2.0m) and LEFT (y=-3.0m)
    - Action: set_attitude(dpitch=+8, droll=0, dyaw=-12)  # Look up and turn left
    
    Common attitude adjustments:
    - Target above (z<0): dpitch>0 (pitch up, nose up)
    - Target below (z>0): dpitch<0 (pitch down, nose down)
    - Target left (y<0): dyaw<0 (turn left)
    - Target right (y>0): dyaw>0 (turn right)
    
    ⚠️ ATTITUDE VERIFICATION:
    Before calling, verify your logic:
    - If target z=+5.29m (below), should I pitch down? YES! (dpitch<0)
    - If target y=-3.0m (left), should I turn left? YES! (dyaw<0)
    - If target z=-1.8m (above), should I pitch up? YES! (dpitch>0)
    
    Args:
        dpitch: Pitch angle change, unit: degrees, positive for nose up, negative for nose down
        droll: Roll angle change, unit: degrees, positive for right roll, negative for left roll
        dyaw: Yaw angle change, unit: degrees, positive for right turn, negative for left turn
    
    Returns:
        Operation result description string
    """
    try:    
        # Connect to Redis
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
        redis_client.ping()
        
        # Convert degrees to radians
        dpitch_rad = math.radians(dpitch)
        droll_rad = math.radians(droll)
        dyaw_rad = math.radians(dyaw)

        # Publish attitude change (position parameters are 0)
        pose_change = {
            'dx': 0.0,
            'dy': 0.0,
            'dz': 0.0,
            'dpitch': round(dpitch_rad, 3),
            'droll': round(droll_rad, 3),
            'dyaw': round(dyaw_rad, 3),
            'timestamp': str(int(time.time() * 1e9))
        }
        
        message = json.dumps(pose_change)
        redis_client.publish('topic.pose_change', message)
        redis_client.close()
        
        return f"Attitude change completed: dpitch={dpitch}°, droll={droll}°, dyaw={dyaw}°"
        
    except Exception as e:
        return f"Failed to set attitude: {str(e)}"

@mcp.tool()
def terminate_navigation(reason: str = "Task completed") -> str:
    """Terminate navigation task
    
    Call this tool to terminate navigation when the task is completed, 
    dangerous situations are encountered, or execution cannot continue.
    
    Applicable scenarios:
    - Successfully approached target spacecraft to specified distance
    - Task objectives completed (such as maintenance, inspection, etc.)
    - Dangerous situations requiring immediate stop
    - Target not found after multiple exploration attempts
    
    Args:
        reason: Specific reason for terminating navigation, used for logging and analysis
    
    Returns:
        Termination operation confirmation message
    """
    # Note: Do not use print in MCP tools, it will break stdio communication protocol
    return f"Navigation terminated successfully. Reason: {reason}"


@mcp.tool()
def lidar_info() -> str:
    """Get LiDAR point cloud information
    
    Analyze current LiDAR scan point cloud data to calculate the center position 
    and boundary range of the target spacecraft.
    
    Returned information includes:
    - Target spacecraft center point coordinates in LiDAR coordinate system
    - Point cloud boundary range (minimum and maximum values)
    - Total number of points
    - Coordinate system description
    
    LiDAR coordinate system:
    - Origin: LiDAR center
    - X-axis positive direction: forward
    - Y-axis positive direction: right
    - Z-axis positive direction: downward
    
    Use cases:
    - When need to know precise position of target spacecraft
    - When calculating distance to target
    - When planning obstacle avoidance
    
    Returns:
        Formatted string containing center point coordinates, boundary range, point count, etc.
    """
    try:
        # Use socket_timeout to avoid infinite waiting
        redis_client = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=2.0)
        redis_client.ping()  # Test connection
        
        # Get latest LiDAR data from Redis
        lidar_data = redis_client.get('latest_lidar_data')
        # Note: Do not use print in MCP tools, it will break stdio communication protocol
        
        if not lidar_data:
            redis_client.close()
            return "Error: No LiDAR data available. Please ensure AirSim data collector is running."
        
        # Parse LiDAR data - use json instead of ast.literal_eval
        try:
            # Try json parsing first
            try:
                data = json.loads(lidar_data.decode('utf-8'))
            except json.JSONDecodeError:
                # If json parsing fails, try to fix single quote issue
                import ast
                data = ast.literal_eval(lidar_data.decode('utf-8'))
            
            points = data.get('points', [])
        except Exception as e:
            redis_client.close()
            return f"Error: LiDAR data parsing failed: {str(e)}"
        
        if not points:
            redis_client.close()
            return "Error: Point cloud data is empty"
        
        # Convert point cloud data to numpy array
        try:
            # Ensure point count is multiple of 3
            if len(points) % 3 != 0:
                points = points[:len(points) - (len(points) % 3)]
            
            points_array = np.array(points).reshape(-1, 3)
            
            # Calculate center point
            center = np.mean(points_array, axis=0)
            
            # Calculate point cloud range
            min_bounds = np.min(points_array, axis=0)
            max_bounds = np.max(points_array, axis=0)
            
            redis_client.close()
            
            # Format return string
            result_str = f"""LiDAR point cloud information:
Point count: {len(points_array)} points
Target center: x={center[0]:.3f}m, y={center[1]:.3f}m, z={center[2]:.3f}m
Boundary range: 
  Minimum: x={min_bounds[0]:.3f}m, y={min_bounds[1]:.3f}m, z={min_bounds[2]:.3f}m
  Maximum: x={max_bounds[0]:.3f}m, y={max_bounds[1]:.3f}m, z={max_bounds[2]:.3f}m
Coordinate system: LiDAR coordinate system (X-axis forward, Y-axis right, Z-axis downward)"""
            
            return result_str
            
        except Exception as e:
            redis_client.close()
            return f"Error: Point cloud processing failed: {str(e)}"
            
    except redis.ConnectionError:
        return "Error: Cannot connect to Redis server, please ensure Redis is running"
    except redis.TimeoutError:
        return "Error: Redis connection timeout"
    except Exception as e:
        return f"Error: Failed to get LiDAR information: {str(e)}"


@mcp.tool()
def image_bright() -> str:
    """Analyze image brightness quality
    
    Calculate brightness statistics of current camera image to determine 
    if image quality is suitable for visual analysis.
    
    Analysis content:
    - Average brightness of foreground regions (areas brighter than overall mean)
    - Maximum and minimum brightness values of the image
    - Brightness dynamic range
    
    Use cases:
    - Determine if current lighting conditions are suitable for visual analysis
    - Decide if camera exposure parameters need adjustment
    - Evaluate if image quality meets task requirements
    
    Brightness assessment criteria:
    - 0-50: Image too dark, may need increased exposure
    - 50-200: Appropriate brightness, suitable for analysis
    - 200-255: Image too bright, may need reduced exposure
    
    Returns:
        Formatted string containing foreground brightness, max/min brightness, brightness range, etc.
    """
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=2.0)
        redis_client.ping()
        
        # Get latest image data from Redis
        image_data = redis_client.get('latest_image_data')
        
        if not image_data:
            redis_client.close()
            return "Error: No image data available. Please ensure AirSim data collector is running."
    except redis.ConnectionError:
        return "Error: Cannot connect to Redis server, please ensure Redis is running"
    except redis.TimeoutError:
        return "Error: Redis connection timeout"
    except Exception as e:
        return f"Error: Redis connection error: {str(e)}"
    
    # Parse image data
    try:
        data = json.loads(image_data.decode('utf-8'))
    except json.JSONDecodeError:
        # If json parsing fails, try ast.literal_eval
        try:
            import ast
            data = ast.literal_eval(image_data.decode('utf-8'))
        except Exception as e:
            return f"Error: Image data parsing failed: {str(e)}"
    
    img_base64 = data.get('data', '')
    
    # Decode base64 image
    try:
        img_bytes = base64.b64decode(img_base64)
        img = Image.open(BytesIO(img_bytes))
        img_array = np.array(img)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img_array
            
        # Calculate image mean as threshold
        threshold = np.mean(gray)
        
        # Extract foreground regions (areas brighter than mean)
        foreground = gray[gray > threshold]
        
        mean_brightness = np.mean(foreground)
        max_brightness = np.max(gray)
        min_brightness = np.min(gray)
        brightness_range = max_brightness - min_brightness
        
        redis_client.close()
        
        # Brightness assessment
        if mean_brightness < 50:
            brightness_assessment = "Image too dark, recommend increasing exposure"
        elif mean_brightness > 200:
            brightness_assessment = "Image too bright, recommend reducing exposure"
        else:
            brightness_assessment = "Appropriate brightness, suitable for analysis"
        
        # Format return string
        result_str = f"""Image brightness analysis:
Foreground average brightness: {mean_brightness:.1f}
Maximum brightness: {max_brightness:.1f}
Minimum brightness: {min_brightness:.1f}
Brightness dynamic range: {brightness_range:.1f}
Brightness assessment: {brightness_assessment}"""
        
        return result_str
        
    except Exception as e:
        return f"Error: Image brightness analysis failed: {str(e)}"


@mcp.tool()
def part_segmentation() -> dict:
    """Get target spacecraft component segmentation results
    
    Identify different components of target spacecraft based on semantic segmentation model, 
    return segmentation image for large model analysis.
    
    Function description:
    - Identify main spacecraft components (such as solar panels, main body, etc.)
    - Return color segmentation image, different colors represent different components
    - Provide image dimension information
    
    Use cases:
    - When need to identify specific components for maintenance
    - When analyzing spacecraft structural integrity
    - When planning approach path to avoid sensitive components
    
    Notes:
    - Returned segmentation image needs to be interpreted with vision model
    - Segmentation accuracy depends on current image quality and distance
    - Recommend using at appropriate distance and lighting conditions
    
    Returns:
        Dictionary containing base64 encoded segmentation image and image dimension information
    """
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=2.0)
        redis_client.ping()
        
        # Get latest segmentation image data from Redis
        seg_data = redis_client.get('latest_segmentation_data')
        
        if not seg_data:
            redis_client.close()
            return {"error": "No segmentation data available. Please ensure AirSim data collector is running."}
    except redis.ConnectionError:
        return {"error": "Cannot connect to Redis server, please ensure Redis is running"}
    except redis.TimeoutError:
        return {"error": "Redis connection timeout"}
    except Exception as e:
        return {"error": f"Redis connection error: {str(e)}"}
    
    # Parse segmentation data
    try:
        data = json.loads(seg_data.decode('utf-8'))
    except json.JSONDecodeError:
        # If json parsing fails, try ast.literal_eval
        data = ast.literal_eval(seg_data.decode('utf-8'))
    img_base64 = data.get('data', '')
    
    # Decode base64 image
    img_bytes = base64.b64decode(img_base64)
    img = Image.open(BytesIO(img_bytes))
    
    # Directly encode segmentation image as base64 string for large model
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    redis_client.close()
    
    return {
        "segmentation_image": img_str,  # Directly pass segmentation image to large model for understanding
        "image_width": img.width,
        "image_height": img.height,
        "message": "Segmentation image is ready, can be directly passed to large model for analysis"
    }


@mcp.tool()
def pose_estimation() -> str:
    """Estimate relative pose of target spacecraft
    
    Estimate position and attitude information of target spacecraft relative to service spacecraft 
    based on vision algorithms.
    
    Estimation content:
    - Relative position of target spacecraft (x, y, z coordinates)
    - Relative attitude of target spacecraft (pitch, roll, yaw angles)
    - Estimation accuracy and confidence information
    
    Use cases:
    - When need to precisely approach target spacecraft
    - Before docking or grasping operations
    - When evaluating relative motion state
    
    Coordinate system description:
    - Consistent with service spacecraft body coordinate system
    - X-axis: forward, Y-axis: right, Z-axis: downward
    
    Notes:
    - Estimation accuracy affected by distance, lighting, target features, etc.
    - Recommend combining with LiDAR information for verification
    - Current version based on simulation ground truth, real deployment requires actual algorithms
    
    Returns:
        Formatted string containing relative pose, distance, azimuth, elevation, etc.
    """   
    try:
        # Connect to Redis to get pose ground truth data
        redis_client = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=2.0)
        redis_client.ping()
        
        # Get latest pose ground truth
        pose_data = redis_client.get('latest_pose_truth')
        
        if not pose_data:
            redis_client.close()
            return "Error: No pose estimation data available. Please ensure AirSim data collector is running."
        
        # Parse pose data
        try:
            pose_truth = json.loads(pose_data.decode('utf-8'))
        except json.JSONDecodeError:
            import ast
            pose_truth = ast.literal_eval(pose_data.decode('utf-8'))
        
        # Extract key information
        relative_pos = pose_truth.get('relative_position', {})
        distance = pose_truth.get('distance', 0)
        azimuth = pose_truth.get('azimuth_deg', 0)
        elevation = pose_truth.get('elevation_deg', 0)
        
        # Add pose assessment information
        if distance < 2:
            proximity_status = "Very close to target (< 2 meters)"
        elif distance < 5:
            proximity_status = "Close to target (2-5 meters)"
        elif distance < 10:
            proximity_status = "Medium distance (5-10 meters)"
        else:
            proximity_status = "Far distance (> 10 meters)"
        
        # Format return string
        result_str = f"""Pose estimation results:
Distance: {distance:.2f} meters
Relative position: x={relative_pos.get('x', 0):.2f}m, y={relative_pos.get('y', 0):.2f}m, z={relative_pos.get('z', 0):.2f}m
Direction angles: azimuth={azimuth:.1f}°, elevation={elevation:.1f}°
Proximity status: {proximity_status}
Coordinate system: Position and attitude of target spacecraft in service spacecraft body coordinate system (X-axis forward, Y-axis right, Z-axis downward)
Estimation method: Based on AirSim simulation ground truth"""
        
        redis_client.close()
        return result_str
        
    except redis.ConnectionError:
        return "Error: Cannot connect to Redis server, please ensure Redis is running"
    except redis.TimeoutError:
        return "Error: Redis connection timeout"
    except Exception as e:
        return f"Error: Error occurred while getting pose estimation: {str(e)}"


@mcp.tool()
def knowledge_base() -> str:
    """Get task-related knowledge base information
    
    Provide professional knowledge and parameter information required for executing 
    on-orbit service tasks.
    
    Knowledge base content:
    - Physical parameters of target spacecraft (dimensions, weight, etc.)
    - Sensor parameters (camera field of view, LiDAR range, etc.)
    - Type and orbit of target spacecraft
    
    Use cases:
    - When need to understand target spacecraft specifications
    - When judging current status based on sensor parameters
    
    Knowledge sources:
    - Spacecraft design documents, sensor technical specifications, 
      on-orbit service operation manuals, safety operation procedures
    
    Returns:
        Formatted string containing relevant knowledge information
    """
    # Knowledge base content
    knowledge_content = """Task-related knowledge base:

Target spacecraft parameters:
- Type: Single-wing solar panel satellite
- Main components: Solar panels, main structure
- Name: EO-1

"""

    return knowledge_content



# Run the server
if __name__ == "__main__":
    mcp.run(transport="stdio")
