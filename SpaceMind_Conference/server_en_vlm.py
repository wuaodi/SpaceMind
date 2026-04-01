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


# Run the server
if __name__ == "__main__":
    mcp.run(transport="stdio")
