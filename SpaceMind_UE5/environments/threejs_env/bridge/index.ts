// Usage:
//   1) Start Redis first
//   2) In the threejs_env directory run: cmd /c npm install
//   3) Backend bridge: cmd /c npm run dev:bridge
//   4) Browser frontend: cmd /c npm run dev:app
//   5) Open http://127.0.0.1:5173 , then run host.py as usual
//
// CLI overrides (aligned with fly_redis.py, usually forwarded by launch_env.py):
//   --satellite CAPSTONE            target satellite (BioSentinel/CAPSTONE/Huygens/IBEX/New_Horizons)
//   --init_x -11 --init_y 0 --init_z 0 --init_yaw 0    servicer initial pose
//   --spin_deg_s 0.5                target spin rate (E1 spinning target)
//   --noise --noise_pos 0.1 --noise_att 0.02           E3N actuation noise
//   --fault_axis dy --fault_scale 0.5                  E3F thruster fault
//   --lidar_dropout 0.3                                E4L intermittent LiDAR dropout
//   --exposure_disturb_step 3 --exposure_disturb_value -3   E4E exposure disturbance
//
// Overview:
//   This file is responsible for three things:
//   - Subscribing to the existing Redis control topics and driving the simulation core
//   - Requesting renders from the browser and writing back latest_image_data / latest_pose_truth / latest_lidar_data
//   - Receiving FBX surface sample points from the browser for LiDAR generation

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createClient } from "redis";
import { WebSocketServer } from "ws";
import { RelativeOrbitSim } from "../sim_core/RelativeOrbitSim.ts";
import type {
  AnnotationRoot,
  AppToBridgeMessage,
  BridgeToAppMessage,
  CaptureRequestMessage,
  CaptureResponseMessage,
  ExposureCommand,
  ImageMessage,
  SceneConfig,
  SegmentationMessage,
  StateSnapshotMessage,
} from "../shared/protocol.ts";
import { timestampNs } from "../shared/math.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");
const sceneConfigPath = path.resolve(projectRoot, "scenes", "default_scene.json");
const annotationPath = path.resolve(projectRoot, "annotations", "default_target.json");
const logDir = path.resolve(projectRoot, "logs");

type PendingCapture = {
  resolve: (msg: CaptureResponseMessage | null) => void;
  timer: NodeJS.Timeout;
};

function readJson<T>(filePath: string): T {
  return JSON.parse(fs.readFileSync(filePath, "utf-8")) as T;
}

function parseCliValue(name: string): string | null {
  const index = process.argv.indexOf(`--${name}`);
  if (index < 0 || index + 1 >= process.argv.length) return null;
  return process.argv[index + 1];
}

function parseCliNumber(name: string): number | null {
  const raw = parseCliValue(name);
  if (raw === null) return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

function hasCliFlag(name: string): boolean {
  return process.argv.includes(`--${name}`);
}

// Apply CLI overrides onto the scene config, matching fly_redis.py semantics
function applyCliOverrides(config: SceneConfig): SceneConfig {
  const satellite = parseCliValue("satellite");
  if (satellite) {
    if (!config.satellites[satellite]) {
      throw new Error(`Unknown satellite '${satellite}'. Available: ${Object.keys(config.satellites).join(", ")}`);
    }
    config.target.name = satellite;
  }

  const initX = parseCliNumber("init_x");
  const initY = parseCliNumber("init_y");
  const initZ = parseCliNumber("init_z");
  const initYaw = parseCliNumber("init_yaw");
  if (initX !== null) config.servicer.initial_state.position_world_m.x = initX;
  if (initY !== null) config.servicer.initial_state.position_world_m.y = initY;
  if (initZ !== null) config.servicer.initial_state.position_world_m.z = initZ;
  if (initYaw !== null) config.servicer.initial_state.attitude_rpy_rad.yaw = initYaw;

  const spin = parseCliNumber("spin_deg_s");
  if (spin !== null) config.target.spin_deg_s = spin;

  if (hasCliFlag("noise")) config.noise.enabled = true;
  const noisePos = parseCliNumber("noise_pos");
  const noiseAtt = parseCliNumber("noise_att");
  if (noisePos !== null) config.noise.position_sigma_m = noisePos;
  if (noiseAtt !== null) config.noise.attitude_sigma_rad = noiseAtt;

  const faultAxis = parseCliValue("fault_axis");
  if (faultAxis === "dx" || faultAxis === "dy" || faultAxis === "dz") config.injection.fault_axis = faultAxis;
  const faultScale = parseCliNumber("fault_scale");
  if (faultScale !== null) config.injection.fault_scale = faultScale;
  const lidarDropout = parseCliNumber("lidar_dropout");
  if (lidarDropout !== null) config.injection.lidar_dropout = lidarDropout;
  const disturbStep = parseCliNumber("exposure_disturb_step");
  if (disturbStep !== null) config.injection.exposure_disturb_step = disturbStep;
  const disturbValue = parseCliNumber("exposure_disturb_value");
  if (disturbValue !== null) config.injection.exposure_disturb_value = disturbValue;

  return config;
}

const sceneConfig = applyCliOverrides(readJson<SceneConfig>(sceneConfigPath));
const annotations = readJson<AnnotationRoot>(annotationPath);
const sim = new RelativeOrbitSim(sceneConfig, annotations, logDir);

const TOPIC_POSE = "topic.pose";
const TOPIC_POSE_CHANGE = "topic.pose_change";
const TOPIC_EXPOSURE = "topic.exposure";
const TOPIC_IMAGE = "topic.img";

const KEY_LATEST_IMAGE = "latest_image_data";
const KEY_LATEST_SEGMENTATION = "latest_segmentation_data";
const KEY_LATEST_LIDAR = "latest_lidar_data";
const KEY_LATEST_POSE_TRUTH = "latest_pose_truth";
const SENSOR_CACHE_KEYS = [
  KEY_LATEST_IMAGE,
  KEY_LATEST_SEGMENTATION,
  KEY_LATEST_LIDAR,
  KEY_LATEST_POSE_TRUTH,
] as const;

let browserSocket: import("ws").WebSocket | null = null;
const pendingCaptures = new Map<string, PendingCapture>();
let captureInFlight = false;
let queuedCaptureReason = "";
// E4E: inject a one-shot exposure jump after the N-th pose_change
let poseChangeCount = 0;
let exposureDisturbFired = false;

function sendToBrowser(message: BridgeToAppMessage): void {
  if (!browserSocket || browserSocket.readyState !== browserSocket.OPEN) {
    return;
  }
  browserSocket.send(JSON.stringify(message));
}

async function publishJson(redisClient: ReturnType<typeof createClient>, key: string, payload: unknown): Promise<void> {
  await redisClient.set(key, JSON.stringify(payload));
}

function buildImageMessage(base64Png: string, width: number, height: number, timestamp: string): ImageMessage {
  return {
    name: `capture_${timestamp}.png`,
    timestamp,
    width,
    height,
    data: base64Png,
  };
}

async function requestBrowserCapture(reason: string): Promise<CaptureResponseMessage | null> {
  if (!browserSocket || browserSocket.readyState !== browserSocket.OPEN) {
    return null;
  }

  const captureId = `capture_${timestampNs()}`;
  const message: CaptureRequestMessage = {
    type: "capture_request",
    capture_id: captureId,
    width: sceneConfig.capture_width,
    height: sceneConfig.capture_height,
    reason,
    render_state: sim.buildRenderState(),
  };

  const responsePromise = new Promise<CaptureResponseMessage | null>((resolve) => {
    const timer = setTimeout(() => {
      pendingCaptures.delete(captureId);
      resolve(null);
    }, 2500);
    pendingCaptures.set(captureId, { resolve, timer });
  });

  sendToBrowser(message);
  return responsePromise;
}

async function publishSensorSnapshot(redisClient: ReturnType<typeof createClient>, reason: string): Promise<void> {
  const poseTruth = sim.buildPoseTruthMessage();
  const lidar = sim.buildLidarMessage();
  await publishJson(redisClient, KEY_LATEST_POSE_TRUTH, poseTruth);
  await publishJson(redisClient, KEY_LATEST_LIDAR, lidar);

  const captureResponse = await requestBrowserCapture(reason);
  if (!captureResponse) {
    sim.recordEvent("sensor_publish_skipped", { reason, cause: "browser_not_ready_or_timeout" });
    return;
  }

  sim.setLatestMetrics(captureResponse.metrics);
  const timestamp = timestampNs();
  const imageMessage = buildImageMessage(
    captureResponse.rgb_png_base64,
    captureResponse.width,
    captureResponse.height,
    timestamp,
  );
  await redisClient.publish(TOPIC_IMAGE, JSON.stringify(imageMessage));
  await publishJson(redisClient, KEY_LATEST_IMAGE, imageMessage);

  if (captureResponse.segmentation_png_base64) {
    const segMessage: SegmentationMessage = buildImageMessage(
      captureResponse.segmentation_png_base64,
      captureResponse.width,
      captureResponse.height,
      timestamp,
    );
    await publishJson(redisClient, KEY_LATEST_SEGMENTATION, segMessage);
  }

  const truthSnapshot = sim.buildHiddenTruthSnapshot();
  sim.recordEvent("sensor_published", {
    reason,
    image_timestamp: imageMessage.timestamp,
    visibility_ratio: captureResponse.metrics.visibility_ratio,
    viewpoint_hits: truthSnapshot.viewpoint_hits,
    service_site_hits: truthSnapshot.service_site_hits,
  });
}

async function scheduleCapture(redisClient: ReturnType<typeof createClient>, reason: string): Promise<void> {
  if (captureInFlight) {
    queuedCaptureReason = reason;
    return;
  }

  captureInFlight = true;
  try {
    await publishSensorSnapshot(redisClient, reason);
  } finally {
    captureInFlight = false;
    if (queuedCaptureReason) {
      const nextReason = queuedCaptureReason;
      queuedCaptureReason = "";
      await scheduleCapture(redisClient, nextReason);
    }
  }
}

function parseJsonMessage<T>(raw: unknown): T | null {
  if (typeof raw !== "string") {
    return null;
  }
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

async function start(): Promise<void> {
  fs.mkdirSync(logDir, { recursive: true });

  const redisPublisher = createClient({ url: "redis://127.0.0.1:6379" });
  const redisSubscriber = createClient({ url: "redis://127.0.0.1:6379" });
  await redisPublisher.connect();
  await redisSubscriber.connect();

  await redisPublisher.del([...SENSOR_CACHE_KEYS]);

  const wss = new WebSocketServer({ port: 8765 });
  console.log("[threejs_env] WebSocket bridge listening on ws://127.0.0.1:8765");
  console.log(
    `[threejs_env] target=${sceneConfig.target.name} spin=${sceneConfig.target.spin_deg_s}deg/s ` +
    `init=(${sceneConfig.servicer.initial_state.position_world_m.x}, ` +
    `${sceneConfig.servicer.initial_state.position_world_m.y}, ` +
    `${sceneConfig.servicer.initial_state.position_world_m.z}) ` +
    `noise=${sceneConfig.noise.enabled} fault_axis=${sceneConfig.injection.fault_axis || "none"} ` +
    `lidar_dropout=${sceneConfig.injection.lidar_dropout}`,
  );

  wss.on("connection", (socket) => {
    browserSocket = socket;
    console.log("[threejs_env] Browser connected");
    sendToBrowser({ type: "scene_init", scene_config: sceneConfig });
    sendToBrowser({ type: "state_snapshot", render_state: sim.buildRenderState() });

    socket.on("message", (buffer) => {
      const message = parseJsonMessage<AppToBridgeMessage>(buffer.toString("utf-8"));
      if (!message) {
        return;
      }

      if (message.type === "ready") {
        sendToBrowser({ type: "scene_init", scene_config: sceneConfig });
        return;
      }

      if (message.type === "model_info") {
        sim.setTargetModelInfo(message.surface_points_body, message.bounding_radius_m);
        console.log(
          `[threejs_env] Target model loaded: ${message.target_name}, ` +
          `${Math.floor(message.surface_points_body.length / 3)} surface points, ` +
          `parts: ${message.part_names.slice(0, 8).join(", ")}${message.part_names.length > 8 ? ", ..." : ""}`,
        );
        return;
      }

      if (message.type === "capture_response") {
        const pending = pendingCaptures.get(message.capture_id);
        if (!pending) {
          return;
        }
        clearTimeout(pending.timer);
        pendingCaptures.delete(message.capture_id);
        pending.resolve(message);
        return;
      }

      if (message.type === "error") {
        console.error("[threejs_env] Browser error:", message.message);
      }
    });

    socket.on("close", () => {
      if (browserSocket === socket) {
        browserSocket = null;
      }
      console.log("[threejs_env] Browser disconnected");
    });
  });

  await redisSubscriber.subscribe(TOPIC_POSE, async (payload) => {
    const command = parseJsonMessage<Record<string, unknown>>(payload);
    if (!command) return;
    sim.recordEvent("command_received", { topic: TOPIC_POSE, command });
    sim.applyAbsolutePose(command);
    await scheduleCapture(redisPublisher, "absolute_pose");
  });

  await redisSubscriber.subscribe(TOPIC_POSE_CHANGE, async (payload) => {
    const command = parseJsonMessage<Record<string, number | string>>(payload);
    if (!command) return;
    sim.recordEvent("command_received", { topic: TOPIC_POSE_CHANGE, command });
    sim.applyDeltaPose(command);

    poseChangeCount += 1;
    const disturbStep = sceneConfig.injection.exposure_disturb_step;
    if (disturbStep > 0 && !exposureDisturbFired && poseChangeCount >= disturbStep) {
      exposureDisturbFired = true;
      sim.setExposure(sceneConfig.injection.exposure_disturb_value);
      sim.recordEvent("exposure_disturb_injected", {
        after_pose_change: poseChangeCount,
        exposure_value: sceneConfig.injection.exposure_disturb_value,
      });
      console.log(`[threejs_env] Exposure disturbance injected: ${sceneConfig.injection.exposure_disturb_value}`);
    }

    await scheduleCapture(redisPublisher, "delta_pose");
  });

  await redisSubscriber.subscribe(TOPIC_EXPOSURE, async (payload) => {
    const command = parseJsonMessage<ExposureCommand>(payload);
    if (!command) return;
    sim.recordEvent("command_received", { topic: TOPIC_EXPOSURE, command });
    sim.setExposure(command.exposure_value);
    await scheduleCapture(redisPublisher, "exposure_change");
  });

  const simStepMs = Math.max(20, Math.round(1000 / sceneConfig.sim_hz));
  const publishMs = Math.max(200, Math.round(1000 / sceneConfig.publish_hz));
  setInterval(() => {
    sim.step(simStepMs / 1000);
    const snapshot: StateSnapshotMessage = { type: "state_snapshot", render_state: sim.buildRenderState() };
    sendToBrowser(snapshot);
  }, simStepMs);

  setInterval(() => {
    void scheduleCapture(redisPublisher, "periodic");
  }, publishMs);

  console.log(`[threejs_env] Redis topics ready: ${TOPIC_POSE}, ${TOPIC_POSE_CHANGE}, ${TOPIC_EXPOSURE}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  start().catch((error) => {
    console.error("[threejs_env] Bridge failed:", error);
    process.exitCode = 1;
  });
}

export { start };
