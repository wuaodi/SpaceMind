// Usage:
//   1) In the threejs_env directory run: cmd /c npm install
//   2) Start the frontend: cmd /c npm run dev:app
//   3) Keep this page open; the bridge backend requests renders and captures over WebSocket.
//
// Overview:
//   This page renders the space scene: it loads the target satellite FBX model
//   (scaled to its real maximum diameter), renders RGB / segmentation frames on
//   capture_request, and after the model loads it uniformly samples the surface
//   and sends the body-frame point cloud back to the bridge for LiDAR generation.

import "./styles.css";
import * as THREE from "three";
import { FBXLoader } from "three/addons/loaders/FBXLoader.js";
import { MeshSurfaceSampler } from "three/addons/math/MeshSurfaceSampler.js";
import type {
  AppToBridgeMessage,
  BridgeToAppMessage,
  CaptureRequestMessage,
  CaptureResponseMessage,
  RenderState,
  SceneConfig,
  Vec3,
} from "../../shared/protocol.ts";
import { add, bodyAxesInWorld, bodyToWorld, norm, scale, sub } from "../../shared/math.ts";

const canvas = document.querySelector<HTMLCanvasElement>("#scene-canvas");
const statusEl = document.querySelector<HTMLDivElement>("#status");
const telemetryEl = document.querySelector<HTMLDivElement>("#telemetry");
if (!canvas || !statusEl || !telemetryEl) {
  throw new Error("Missing app DOM nodes.");
}

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight, false);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;

const scene = new THREE.Scene();
scene.background = new THREE.Color("#000000");

const overviewCamera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 2000);
// Keep the overview camera on the sunlit side (sun at (-30,-20,-15)); otherwise it
// faces the shadowed side of the target and textures are flattened by ambient light.
overviewCamera.position.set(-14, -10, -7);
overviewCamera.lookAt(0, 0, 0);

const sensorCamera = new THREE.PerspectiveCamera(60, 640 / 480, 0.1, 500);

// Space lighting: directional sunlight as the key light, weak ambient as fill
scene.add(new THREE.AmbientLight(0xffffff, 0.35));
const sun = new THREE.DirectionalLight(0xffffff, 3.0);
sun.position.set(-30, -20, -15);
scene.add(sun);

const starGeometry = new THREE.BufferGeometry();
const starPositions: number[] = [];
const starSizes: number[] = [];
for (let i = 0; i < 2500; i += 1) {
  // Distribute stars on a distant spherical shell so none appear near the target
  const radius = 400 + Math.random() * 400;
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos(2 * Math.random() - 1);
  starPositions.push(
    radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.sin(phi) * Math.sin(theta),
    radius * Math.cos(phi),
  );
  starSizes.push(0.6 + Math.random() * 1.4);
}
starGeometry.setAttribute("position", new THREE.Float32BufferAttribute(starPositions, 3));
const starPoints = new THREE.Points(
  starGeometry,
  new THREE.PointsMaterial({ color: 0xffffff, size: 1.2, sizeAttenuation: true }),
);
scene.add(starPoints);

const servicerGroup = new THREE.Group();
const targetGroup = new THREE.Group();
scene.add(servicerGroup);
scene.add(targetGroup);

const targetMeshes: THREE.Mesh[] = [];
const segmentationMaterials = new Map<THREE.Mesh, THREE.Material>();
const targetReferencePoints: THREE.Vector3[] = [];

let latestState: RenderState | null = null;
let loadedModelUrl = "";
let ws: WebSocket | null = null;

const SEGMENTATION_PALETTE = [
  "#ff4444", "#44ff44", "#4488ff", "#ffee44", "#ff44ff",
  "#44ffff", "#ff8844", "#88ff44", "#8844ff", "#ff4488",
  "#44ff88", "#4844ff", "#ffaa88", "#aaff88", "#88aaff",
];

function toThree(v: Vec3): THREE.Vector3 {
  return new THREE.Vector3(v.x, v.y, v.z);
}

function buildServicer(config: SceneConfig) {
  while (servicerGroup.children.length > 0) {
    servicerGroup.remove(servicerGroup.children[0]);
  }
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(config.servicer.body_size_m.x, config.servicer.body_size_m.y, config.servicer.body_size_m.z),
    new THREE.MeshStandardMaterial({ color: "#7ab8ff", roughness: 0.5, metalness: 0.4 }),
  );
  servicerGroup.add(body);
  const cameraMarker = new THREE.Mesh(
    new THREE.ConeGeometry(0.18, 0.5, 18),
    new THREE.MeshStandardMaterial({ color: "#ffc260" }),
  );
  cameraMarker.rotation.z = -Math.PI / 2;
  cameraMarker.position.set(config.camera_offset_body_m.x + 0.35, 0, 0);
  servicerGroup.add(cameraMarker);
}

// Uniformly sample the surface of each mesh; returns a flattened body-frame point list
function sampleModelSurface(meshes: THREE.Mesh[], totalPoints: number): number[] {
  if (meshes.length === 0) return [];
  const perMesh = Math.max(20, Math.ceil(totalPoints / meshes.length));
  const flat: number[] = [];
  const position = new THREE.Vector3();
  for (const mesh of meshes) {
    let sampler: MeshSurfaceSampler;
    try {
      sampler = new MeshSurfaceSampler(mesh).build();
    } catch {
      continue;
    }
    for (let i = 0; i < perMesh; i += 1) {
      sampler.sample(position);
      position.applyMatrix4(mesh.matrixWorld);
      flat.push(
        Math.round(position.x * 1000) / 1000,
        Math.round(position.y * 1000) / 1000,
        Math.round(position.z * 1000) / 1000,
      );
    }
  }
  return flat;
}

function loadTargetModel(modelUrl: string, maxDiameterM: number, targetName: string) {
  if (loadedModelUrl === modelUrl) return;
  loadedModelUrl = modelUrl;

  const loader = new FBXLoader();
  statusEl.textContent = `Loading model: ${targetName} ...`;
  loader.load(
    modelUrl,
    (model) => {
      while (targetGroup.children.length > 0) {
        targetGroup.remove(targetGroup.children[0]);
      }
      targetMeshes.length = 0;
      targetReferencePoints.length = 0;
      segmentationMaterials.clear();

      // Scale to the real maximum diameter and move the bounding-box center to the origin
      const rawBox = new THREE.Box3().setFromObject(model);
      const rawSize = rawBox.getSize(new THREE.Vector3());
      const rawMax = Math.max(rawSize.x, rawSize.y, rawSize.z);
      const scaleFactor = rawMax > 1e-9 ? maxDiameterM / rawMax : 1.0;
      model.scale.setScalar(scaleFactor);

      const scaledBox = new THREE.Box3().setFromObject(model);
      const center = scaledBox.getCenter(new THREE.Vector3());
      model.position.sub(center);

      targetGroup.add(model);
      targetGroup.updateMatrixWorld(true);

      let partIndex = 0;
      const partNames: string[] = [];
      model.traverse((child) => {
        if (!(child as THREE.Mesh).isMesh) return;
        const mesh = child as THREE.Mesh;
        targetMeshes.push(mesh);
        const partName = mesh.name || `part_${partIndex}`;
        partNames.push(partName);
        segmentationMaterials.set(
          mesh,
          new THREE.MeshBasicMaterial({ color: SEGMENTATION_PALETTE[partIndex % SEGMENTATION_PALETTE.length] }),
        );
        partIndex += 1;
      });

      // Bounding-box corners are used for the visibility metric
      const finalBox = new THREE.Box3().setFromObject(targetGroup);
      for (const sx of [finalBox.min.x, finalBox.max.x]) {
        for (const sy of [finalBox.min.y, finalBox.max.y]) {
          for (const sz of [finalBox.min.z, finalBox.max.z]) {
            targetReferencePoints.push(new THREE.Vector3(sx, sy, sz));
          }
        }
      }

      // Send surface samples back to the bridge (targetGroup is at the origin with no
      // rotation here, so world frame equals body frame)
      const surfacePoints = sampleModelSurface(targetMeshes, 600);
      const boundingRadius = finalBox.getSize(new THREE.Vector3()).length() / 2;
      sendMessage({
        type: "model_info",
        target_name: targetName,
        bounding_radius_m: Math.round(boundingRadius * 1000) / 1000,
        surface_points_body: surfacePoints,
        part_names: partNames,
      });

      statusEl.textContent = `Model loaded: ${targetName} (${partNames.length} parts)`;
      if (latestState) {
        applyRenderState(latestState);
      }
    },
    undefined,
    (error) => {
      statusEl.textContent = `Model load failed: ${targetName}`;
      sendMessage({ type: "error", message: `FBX load failed for ${modelUrl}: ${String(error)}` });
    },
  );
}

function applyBodyTransform(object: THREE.Object3D, position: Vec3, attitude: { pitch: number; roll: number; yaw: number }) {
  const axes = bodyAxesInWorld(attitude);
  object.position.set(position.x, position.y, position.z);
  // Body frame: x forward, y right, z down (right-handed). The columns are the body
  // axes expressed in the world frame, matching rotationMatrixFromEuler in shared/math.ts.
  // Do not pass a left-handed basis (e.g. right/up/forward): makeBasis would produce a
  // reflection matrix and the extracted rotation would be wrong.
  const forward = new THREE.Vector3(axes.forward.x, axes.forward.y, axes.forward.z);
  const right = new THREE.Vector3(axes.right.x, axes.right.y, axes.right.z);
  const down = new THREE.Vector3(axes.down.x, axes.down.y, axes.down.z);
  const basis = new THREE.Matrix4().makeBasis(forward, right, down);
  object.setRotationFromMatrix(basis);
}

function applyRenderState(state: RenderState) {
  latestState = state;
  loadTargetModel(state.target.model_url, state.target.max_diameter_m, state.target_name);
  applyBodyTransform(servicerGroup, state.servicer.position_world_m, state.servicer.attitude_rpy_rad);
  applyBodyTransform(targetGroup, state.target.position_world_m, state.target.attitude_rpy_rad);

  const cameraOffsetWorld = bodyToWorld(state.camera_offset_body_m, state.servicer.attitude_rpy_rad);
  const cameraPosition = add(state.servicer.position_world_m, cameraOffsetWorld);
  const axes = bodyAxesInWorld(state.servicer.attitude_rpy_rad);
  const forwardPoint = add(cameraPosition, scale(axes.forward, 5));
  sensorCamera.position.set(cameraPosition.x, cameraPosition.y, cameraPosition.z);
  sensorCamera.up.set(axes.up.x, axes.up.y, axes.up.z);
  sensorCamera.lookAt(forwardPoint.x, forwardPoint.y, forwardPoint.z);
  sensorCamera.updateProjectionMatrix();

  renderer.toneMappingExposure = Math.pow(2, state.exposure_value);
}

function projectTargetMetrics(width: number, height: number) {
  const points2D: THREE.Vector3[] = [];
  for (const point of targetReferencePoints) {
    const projected = point.clone().applyMatrix4(targetGroup.matrixWorld).project(sensorCamera);
    if (projected.z > -1 && projected.z < 1) {
      points2D.push(projected);
    }
  }

  if (points2D.length === 0) {
    return {
      target_visible: false,
      visibility_ratio: 0,
      projected_bounds_px: null,
    };
  }

  const xs = points2D.map((p) => ((p.x + 1) / 2) * width);
  const ys = points2D.map((p) => ((1 - p.y) / 2) * height);
  const minX = Math.max(0, Math.min(...xs));
  const maxX = Math.min(width, Math.max(...xs));
  const minY = Math.max(0, Math.min(...ys));
  const maxY = Math.min(height, Math.max(...ys));
  const area = Math.max(0, maxX - minX) * Math.max(0, maxY - minY);
  const total = width * height;
  return {
    target_visible: area > 1,
    visibility_ratio: total > 0 ? area / total : 0,
    projected_bounds_px: {
      min_x: Math.round(minX),
      min_y: Math.round(minY),
      max_x: Math.round(maxX),
      max_y: Math.round(maxY),
    },
  };
}

// Part-level segmentation: render one frame with each sub-mesh in a solid color
function renderSegmentationPass() {
  const originalMaterials = new Map<THREE.Mesh, THREE.Material | THREE.Material[]>();
  for (const mesh of targetMeshes) {
    originalMaterials.set(mesh, mesh.material);
    mesh.material = segmentationMaterials.get(mesh) ?? new THREE.MeshBasicMaterial({ color: "#ffffff" });
  }

  starPoints.visible = false;
  const originalToneMapping = renderer.toneMapping;
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.render(scene, sensorCamera);
  const base64 = canvas.toDataURL("image/png").replace(/^data:image\/png;base64,/, "");
  renderer.toneMapping = originalToneMapping;
  starPoints.visible = true;

  for (const mesh of targetMeshes) {
    const original = originalMaterials.get(mesh);
    if (original) {
      mesh.material = original;
    }
  }
  return base64;
}

function captureFrame(request: CaptureRequestMessage): CaptureResponseMessage {
  applyRenderState(request.render_state);
  renderer.setSize(request.width, request.height, false);
  sensorCamera.aspect = request.width / request.height;
  sensorCamera.updateProjectionMatrix();

  servicerGroup.visible = false;
  renderer.render(scene, sensorCamera);
  const rgbBase64 = canvas.toDataURL("image/png").replace(/^data:image\/png;base64,/, "");
  const segmentationBase64 = renderSegmentationPass();
  servicerGroup.visible = true;

  const metrics = projectTargetMetrics(request.width, request.height);
  return {
    type: "capture_response",
    capture_id: request.capture_id,
    width: request.width,
    height: request.height,
    rgb_png_base64: rgbBase64,
    segmentation_png_base64: segmentationBase64,
    metrics,
  };
}

function updateTelemetry() {
  if (!latestState) {
    telemetryEl.textContent = "";
    return;
  }
  const servicer = latestState.servicer.position_world_m;
  const target = latestState.target.position_world_m;
  const relative = sub(target, servicer);
  telemetryEl.textContent = [
    `Scene: ${latestState.scene_name}`,
    `Target: ${latestState.target_name}`,
    `Position(x,y,z): ${servicer.x.toFixed(2)}, ${servicer.y.toFixed(2)}, ${servicer.z.toFixed(2)} m`,
    `Attitude(p,r,y): ${latestState.servicer.attitude_rpy_rad.pitch.toFixed(2)}, ${latestState.servicer.attitude_rpy_rad.roll.toFixed(2)}, ${latestState.servicer.attitude_rpy_rad.yaw.toFixed(2)} rad`,
    `Relative vector: ${relative.x.toFixed(2)}, ${relative.y.toFixed(2)}, ${relative.z.toFixed(2)} m`,
    `Distance: ${norm(relative).toFixed(2)} m`,
    `Exposure: ${latestState.exposure_value.toFixed(2)}`
  ].join("\n");
}

function animate() {
  requestAnimationFrame(animate);
  if (latestState) {
    overviewCamera.lookAt(toThree(latestState.target.position_world_m));
  }
  renderer.render(scene, overviewCamera);
}

window.addEventListener("resize", () => {
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  overviewCamera.aspect = window.innerWidth / window.innerHeight;
  overviewCamera.updateProjectionMatrix();
});

function sendMessage(message: AppToBridgeMessage) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(message));
  }
}

ws = new WebSocket("ws://127.0.0.1:8765");
ws.addEventListener("open", () => {
  statusEl.textContent = "Connected to bridge backend";
  sendMessage({ type: "ready" });
});

ws.addEventListener("close", () => {
  statusEl.textContent = "Bridge backend disconnected";
});

ws.addEventListener("message", (event) => {
  const message = JSON.parse(String(event.data)) as BridgeToAppMessage;

  if (message.type === "scene_init") {
    buildServicer(message.scene_config);
    statusEl.textContent = `Scene loaded: ${message.scene_config.scene_name}`;
    return;
  }

  if (message.type === "state_snapshot") {
    applyRenderState(message.render_state);
    updateTelemetry();
    return;
  }

  if (message.type === "capture_request") {
    const response = captureFrame(message);
    ws?.send(JSON.stringify(response));
    return;
  }

  if (message.type === "error") {
    statusEl.textContent = `Bridge error: ${message.message}`;
  }
});

animate();
