import math
import time

from mcp.server.fastmcp import FastMCP

from .common import TOPIC_EXPOSURE, TOPIC_POSE_CHANGE, publish_json_message


def register_env_tools(mcp: FastMCP) -> None:
    @mcp.tool()
    def set_position(dx: float, dy: float, dz: float) -> str:
        """Control spacecraft position change (translation control).

        Controls the service spacecraft position movement via thrusters.
        All changes are increments in the body coordinate system.

        Coordinate system (spacecraft body frame):
        - X-axis: Forward positive (+dx), backward negative (-dx)
        - Y-axis: Right positive (+dy), left negative (-dy)
        - Z-axis: Downward positive (+dz), upward negative (-dz)

        Alignment rule:
        - The relative target coordinates returned by lidar_info() use the SAME body frame.
        - Therefore target x>0 means the target is in front of you, so approaching requires dx>0.
        - target x<0 -> approaching requires dx<0.
        - target y>0 -> dy>0, target y<0 -> dy<0.
        - target z>0 -> dz>0, target z<0 -> dz<0.
        - Negative dx while target x>0 is retreat and increases distance.

        Step sizing rule for approach:
        - Distance-scaled forward motion is mainly for straight front-approach phases with a reliable range cue.
        - For left/right relocation or orbit tasks, do not treat a large forward-only move or a large diagonal jump as the default just because the target is far away.
        - In side-view tasks, translation is used to enter the side sector, and attitude is used to keep the target near the image center.
        - When only visual cues are available, prefer one conservative translation and then re-observe instead of opening with a large move.

        Example: If target is at (x=20m, y=0m, z=5m) in body frame:
        - Target is 20m forward and 5m below
        - To approach: set_position(dx=2.0, dy=0, dz=1.0)  # move forward and down
        - To retreat: set_position(dx=-2.0, dy=0, dz=-1.0)  # move backward and up

        Args:
            dx: X change in meters (positive=forward)
            dy: Y change in meters (positive=right)
            dz: Z change in meters (positive=down)

        Returns:
            Operation result string
        """
        try:
            publish_json_message(
                TOPIC_POSE_CHANGE,
                {
                    "dx": round(dx, 3),
                    "dy": round(dy, 3),
                    "dz": round(dz, 3),
                    "dpitch": 0.0,
                    "droll": 0.0,
                    "dyaw": 0.0,
                    "timestamp": str(int(time.time() * 1e9)),
                },
            )
            return f"Position change sent: dx={dx}m, dy={dy}m, dz={dz}m"
        except Exception as e:
            return f"Failed to set position: {str(e)}"

    @mcp.tool()
    def set_attitude(dpitch: float, droll: float, dyaw: float) -> str:
        """Control spacecraft attitude change (rotation control).

        Controls the service spacecraft attitude via reaction wheels.
        Angles are in degrees; internally converted to radians for Redis.

        Attitude (Euler angles, degrees):
        - dpitch: Pitch change, nose up positive, nose down negative
        - droll: Roll change, right roll positive
        - dyaw: Yaw change, right turn positive

        Example: If target is at (x=20m, y=0m, z=5m) in body frame:
        - Target is below (z>0), need to pitch down: set_attitude(dpitch=-10, droll=0, dyaw=0)
        Example: If target is at (x=15m, y=-3m, z=-2m):
        - Target above and left: set_attitude(dpitch=8, droll=0, dyaw=-12)  # look up and turn left

        Args:
            dpitch: Pitch change in degrees
            droll: Roll change in degrees
            dyaw: Yaw change in degrees

        Returns:
            Operation result string
        """
        try:
            publish_json_message(
                TOPIC_POSE_CHANGE,
                {
                    "dx": 0.0,
                    "dy": 0.0,
                    "dz": 0.0,
                    "dpitch": round(math.radians(dpitch), 3),
                    "droll": round(math.radians(droll), 3),
                    "dyaw": round(math.radians(dyaw), 3),
                    "timestamp": str(int(time.time() * 1e9)),
                },
            )
            return f"Attitude change sent: dpitch={dpitch}°, droll={droll}°, dyaw={dyaw}°"
        except Exception as e:
            return f"Failed to set attitude: {str(e)}"

    @mcp.tool()
    def terminate_navigation(reason: str = "Task completed") -> str:
        """Terminate the navigation task.

        Call when the task is done, a hazard is detected, or execution cannot continue.

        Args:
            reason: Reason for termination (for logging)

        Returns:
            Confirmation message
        """
        return f"Navigation terminated. Reason: {reason}"

    @mcp.tool()
    def set_exposure(exposure_value: float) -> str:
        """Set camera exposure via Redis.

        Publishes to topic.exposure. exposure_value typically in [-3.0, 3.0], 0=default.

        Args:
            exposure_value: Exposure compensation value

        Returns:
            Operation result string
        """
        try:
            publish_json_message(TOPIC_EXPOSURE, {"exposure_value": float(exposure_value)})
            return f"Exposure command sent: {exposure_value}"
        except Exception as e:
            return f"Failed to set exposure: {str(e)}"
