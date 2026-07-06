// Standalone verification script: simulates the bridge without Redis by pushing
// scene_init / state_snapshot to the browser, for manually checking the main page render.
// Run: node fake_bridge.mjs
import fs from "node:fs";
import { WebSocketServer } from "ws";

const sceneConfig = JSON.parse(fs.readFileSync(new URL("./scenes/default_scene.json", import.meta.url), "utf-8"));

const renderState = {
  timestamp: `${Date.now()}000000`,
  scene_name: sceneConfig.scene_name,
  exposure_value: 0,
  target_name: sceneConfig.target.name,
  camera_offset_body_m: sceneConfig.camera_offset_body_m,
  servicer: {
    position_world_m: sceneConfig.servicer.initial_state.position_world_m,
    attitude_rpy_rad: sceneConfig.servicer.initial_state.attitude_rpy_rad,
  },
  target: {
    position_world_m: { x: 0, y: 0, z: 0 },
    attitude_rpy_rad: { pitch: 0, roll: 0, yaw: 0 },
    model_url: sceneConfig.satellites[sceneConfig.target.name].model_url,
    max_diameter_m: sceneConfig.satellites[sceneConfig.target.name].max_diameter_m,
  },
};

const wss = new WebSocketServer({ port: 8765 });
console.log("[fake_bridge] listening on ws://127.0.0.1:8765");
wss.on("connection", (socket) => {
  console.log("[fake_bridge] browser connected");
  socket.send(JSON.stringify({ type: "scene_init", scene_config: sceneConfig }));
  const timer = setInterval(() => {
    socket.send(JSON.stringify({ type: "state_snapshot", render_state: renderState }));
  }, 500);
  // After 8s (textures loaded) request one sensor-camera capture and save it to disk
  setTimeout(() => {
    socket.send(JSON.stringify({
      type: "capture_request",
      capture_id: "verify_1",
      width: 640,
      height: 480,
      reason: "verify",
      render_state: renderState,
    }));
  }, 8000);

  socket.on("message", (buf) => {
    const msg = JSON.parse(buf.toString("utf-8"));
    if (msg.type === "capture_response") {
      fs.writeFileSync("verify_rgb.png", Buffer.from(msg.rgb_png_base64, "base64"));
      fs.writeFileSync("verify_seg.png", Buffer.from(msg.segmentation_png_base64 ?? "", "base64"));
      console.log("[fake_bridge] capture saved, metrics:", JSON.stringify(msg.metrics));
    } else {
      console.log("[fake_bridge] recv:", msg.type);
    }
  });
  socket.on("close", () => clearInterval(timer));
});
