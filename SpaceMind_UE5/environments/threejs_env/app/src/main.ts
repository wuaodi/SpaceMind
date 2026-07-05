// 启动:
//   1) 在 threejs_env 目录执行: cmd /c npm install
//   2) 启动前端: cmd /c npm run dev:app
//   3) 保持本页面打开，桥接后端会通过 WebSocket 请求渲染和截图
//
// 说明:
//   这个页面负责太空场景渲染：加载目标星 FBX 模型（按真实最大直径缩放），
//   在收到 capture_request 时渲染 RGB / Segmentation 并回传；
//   模型加载完成后对表面均匀采样，把体坐标系点云回传给 bridge 供 LiDAR 使用。

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
overviewCamera.position.set(-12, 12, 8);
overviewCamera.lookAt(0, 0, 0);

const sensorCamera = new THREE.PerspectiveCamera(60, 640 / 480, 0.1, 500);

// 太空光照：太阳方向光为主，弱环境光保底
scene.add(new THREE.AmbientLight(0xffffff, 0.35));
const sun = new THREE.DirectionalLight(0xffffff, 3.0);
sun.position.set(-30, -20, -15);
scene.add(sun);

const starGeometry = new THREE.BufferGeometry();
const starPositions: number[] = [];
const starSizes: number[] = [];
for (let i = 0; i < 2500; i += 1) {
  // 星点分布在远球壳上，避免出现在目标附近
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

// 对模型每个 mesh 表面均匀采样，返回体坐标系展平点序列
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
  statusEl.textContent = `正在加载模型: ${targetName} ...`;
  loader.load(
    modelUrl,
    (model) => {
      while (targetGroup.children.length > 0) {
        targetGroup.remove(targetGroup.children[0]);
      }
      targetMeshes.length = 0;
      targetReferencePoints.length = 0;
      segmentationMaterials.clear();

      // 按真实最大直径缩放并把包围盒中心移到原点
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

      // 包围盒角点用于可见性度量
      const finalBox = new THREE.Box3().setFromObject(targetGroup);
      for (const sx of [finalBox.min.x, finalBox.max.x]) {
        for (const sy of [finalBox.min.y, finalBox.max.y]) {
          for (const sz of [finalBox.min.z, finalBox.max.z]) {
            targetReferencePoints.push(new THREE.Vector3(sx, sy, sz));
          }
        }
      }

      // 表面采样点回传 bridge（此时 targetGroup 位于原点且无旋转，世界系即体坐标系）
      const surfacePoints = sampleModelSurface(targetMeshes, 600);
      const boundingRadius = finalBox.getSize(new THREE.Vector3()).length() / 2;
      sendMessage({
        type: "model_info",
        target_name: targetName,
        bounding_radius_m: Math.round(boundingRadius * 1000) / 1000,
        surface_points_body: surfacePoints,
        part_names: partNames,
      });

      statusEl.textContent = `模型已加载: ${targetName} (${partNames.length} parts)`;
      if (latestState) {
        applyRenderState(latestState);
      }
    },
    undefined,
    (error) => {
      statusEl.textContent = `模型加载失败: ${targetName}`;
      sendMessage({ type: "error", message: `FBX load failed for ${modelUrl}: ${String(error)}` });
    },
  );
}

function applyBodyTransform(object: THREE.Object3D, position: Vec3, attitude: { pitch: number; roll: number; yaw: number }) {
  const axes = bodyAxesInWorld(attitude);
  object.position.set(position.x, position.y, position.z);
  const right = new THREE.Vector3(axes.right.x, axes.right.y, axes.right.z);
  const up = new THREE.Vector3(axes.up.x, axes.up.y, axes.up.z);
  const forward = new THREE.Vector3(axes.forward.x, axes.forward.y, axes.forward.z);
  const basis = new THREE.Matrix4().makeBasis(right, up, forward);
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

// 部件级分割：每个子 mesh 换成纯色材质渲染一帧
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
    `场景: ${latestState.scene_name}`,
    `目标: ${latestState.target_name}`,
    `位置(x,y,z): ${servicer.x.toFixed(2)}, ${servicer.y.toFixed(2)}, ${servicer.z.toFixed(2)} m`,
    `姿态(p,r,y): ${latestState.servicer.attitude_rpy_rad.pitch.toFixed(2)}, ${latestState.servicer.attitude_rpy_rad.roll.toFixed(2)}, ${latestState.servicer.attitude_rpy_rad.yaw.toFixed(2)} rad`,
    `相对向量: ${relative.x.toFixed(2)}, ${relative.y.toFixed(2)}, ${relative.z.toFixed(2)} m`,
    `距离: ${norm(relative).toFixed(2)} m`,
    `曝光: ${latestState.exposure_value.toFixed(2)}`
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
  statusEl.textContent = "已连接桥接后端";
  sendMessage({ type: "ready" });
});

ws.addEventListener("close", () => {
  statusEl.textContent = "桥接后端已断开";
});

ws.addEventListener("message", (event) => {
  const message = JSON.parse(String(event.data)) as BridgeToAppMessage;

  if (message.type === "scene_init") {
    buildServicer(message.scene_config);
    statusEl.textContent = `场景已加载: ${message.scene_config.scene_name}`;
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
    statusEl.textContent = `桥接错误: ${message.message}`;
  }
});

animate();
