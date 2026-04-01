import numpy as np
from mcp.server.fastmcp import FastMCP
from PIL import Image

from .common import (
    KEY_LATEST_IMAGE,
    KEY_LATEST_LIDAR,
    KEY_LATEST_SEGMENTATION,
    compute_lidar_surface_distance,
    encode_png_base64,
    load_snapshot_image,
    read_json_snapshot,
)


def register_sensor_tools(mcp: FastMCP) -> None:
    @mcp.tool()
    def lidar_info() -> str:
        """Get LiDAR point cloud information.

        Reads latest_lidar_data from Redis, parses point cloud, computes center and bounds.

        LiDAR frame: X forward, Y right, Z down.

        Returns:
            Center, bounds, point count as formatted string
        """
        try:
            data = read_json_snapshot(KEY_LATEST_LIDAR)
            if not data:
                return "Error: No LiDAR data. Ensure AirSim collector is running."
            points = data.get("points", [])
            if not points or len(points) % 3 != 0:
                return "LiDAR: No point cloud. Target may be out of FOV."
            pts = np.array(points).reshape(-1, 3)
            center = np.mean(pts, axis=0)
            minimum = np.min(pts, axis=0)
            maximum = np.max(pts, axis=0)
            surface_distance = compute_lidar_surface_distance(data)
            surface_text = (
                f"nearest surface distance={surface_distance:.3f}m | "
                if surface_distance is not None
                else ""
            )
            return (
                f"LiDAR: {len(pts)} points | "
                f"{surface_text}"
                f"center x={center[0]:.3f}, y={center[1]:.3f}, z={center[2]:.3f}m | "
                f"min x={minimum[0]:.3f}, y={minimum[1]:.3f}, z={minimum[2]:.3f}m | "
                f"max x={maximum[0]:.3f}, y={maximum[1]:.3f}, z={maximum[2]:.3f}m"
            )
        except Exception as e:
            return f"Error: {str(e)}"

    @mcp.tool()
    def image_bright() -> str:
        """Get image brightness statistics.

        Reads latest_image_data from Redis and computes brightness stats.

        Returns:
            Foreground mean, min, max, range, and assessment
        """
        try:
            snapshot = load_snapshot_image(KEY_LATEST_IMAGE)
            if not snapshot:
                return "Error: No image data. Ensure AirSim collector is running."
            _, image = snapshot
            array = np.array(image)
            gray = np.dot(array[..., :3], [0.2989, 0.5870, 0.1140]) if array.ndim == 3 else array
            threshold = np.mean(gray)
            foreground = gray[gray > threshold]
            mean_brightness = float(np.mean(foreground)) if foreground.size > 0 else float(threshold)
            minimum, maximum = np.min(gray), np.max(gray)
            dynamic_range = maximum - minimum
            if mean_brightness < 50:
                assessment = "Too dark, increase exposure"
            elif mean_brightness > 200:
                assessment = "Too bright, reduce exposure"
            else:
                assessment = "OK for analysis"
            return (
                f"Brightness: mean={mean_brightness:.1f}, min={minimum:.1f}, "
                f"max={maximum:.1f}, range={dynamic_range:.1f} | {assessment}"
            )
        except Exception as e:
            return f"Error: {str(e)}"

    @mcp.tool()
    def part_segmentation() -> dict:
        """Get segmentation image of target spacecraft.

        Reads latest_segmentation_data from Redis, returns base64 segmentation image.

        Returns:
            Dict with segmentation_image (base64), image_width, image_height
        """
        try:
            snapshot = load_snapshot_image(KEY_LATEST_SEGMENTATION)
            if not snapshot:
                return {"error": "No segmentation data. Ensure AirSim collector is running."}
            payload, image = snapshot
            return {
                "segmentation_image": encode_png_base64(image),
                "image_width": payload.get("width", image.width),
                "image_height": payload.get("height", image.height),
            }
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def image_crop(x1: int, y1: int, x2: int, y2: int) -> str:
        """Crop a region from the latest image and return base64.

        Reads latest_image_data from Redis. Pixel coords: (x1,y1)=top-left, (x2,y2)=bottom-right.

        Example: For 640x480 image, crop center 200x200: image_crop(220, 140, 420, 340)

        Args:
            x1: Left x (pixels)
            y1: Top y (pixels)
            x2: Right x (pixels)
            y2: Bottom y (pixels)

        Returns:
            Base64 string of cropped PNG, or error message
        """
        try:
            snapshot = load_snapshot_image(KEY_LATEST_IMAGE)
            if not snapshot:
                return "Error: No image data."
            _, image = snapshot
            array = np.array(image)
            height, width = array.shape[:2]
            left, right = max(0, min(x1, x2)), min(width, max(x1, x2))
            top, bottom = max(0, min(y1, y2)), min(height, max(y1, y2))
            cropped = Image.fromarray(array[top:bottom, left:right])
            return encode_png_base64(cropped)
        except Exception as e:
            return f"Error: {str(e)}"

    @mcp.tool()
    def image_zoom(scale: float) -> str:
        """Zoom the center region of the latest image.

        Reads latest_image_data, crops center 1/scale of image, resizes to original dimensions.
        scale>1: zoom in (magnify center); scale<1: zoom out (shrink and pad).

        Example: image_zoom(2.0) - 2x zoom on center region

        Args:
            scale: Zoom factor (>1 zoom in, <1 zoom out)

        Returns:
            Base64 string of zoomed PNG, or error message
        """
        try:
            snapshot = load_snapshot_image(KEY_LATEST_IMAGE)
            if not snapshot:
                return "Error: No image data."
            _, image = snapshot
            width, height = image.size
            if scale >= 1:
                crop_width = min(width, int(width / scale))
                crop_height = min(height, int(height / scale))
                center_x = (width - crop_width) // 2
                center_y = (height - crop_height) // 2
                crop_box = (
                    max(0, center_x),
                    max(0, center_y),
                    min(width, center_x + crop_width),
                    min(height, center_y + crop_height),
                )
                cropped = image.crop(crop_box)
                zoomed = cropped.resize((width, height), Image.Resampling.LANCZOS)
            else:
                scaled_width = int(width / scale)
                scaled_height = int(height / scale)
                scaled = image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
                center_x = (scaled_width - width) // 2
                center_y = (scaled_height - height) // 2
                crop_box = (
                    max(0, center_x),
                    max(0, center_y),
                    min(scaled_width, center_x + width),
                    min(scaled_height, center_y + height),
                )
                zoomed = scaled.crop(crop_box)
            return encode_png_base64(zoomed)
        except Exception as e:
            return f"Error: {str(e)}"
