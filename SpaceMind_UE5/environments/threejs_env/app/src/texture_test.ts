// Diagnostic page: load an FBX and print the texture status of every material,
// verifying that embedded textures are parsed by FBXLoader.
// Open http://127.0.0.1:5173/texture_test.html?model=CAPSTONE
import * as THREE from "three";
import { FBXLoader } from "three/addons/loaders/FBXLoader.js";

const logEl = document.querySelector<HTMLDivElement>("#log")!;
const lines: string[] = [];
function log(msg: string) {
  lines.push(msg);
  logEl.textContent = lines.join("\n");
  console.log(msg);
}

const params = new URLSearchParams(location.search);
const modelName = params.get("model") ?? "CAPSTONE";
// mode=sim replicates the exact render settings of main.ts (ACES + same lights/camera) for comparison
const simMode = params.get("mode") === "sim";

const canvas = document.querySelector<HTMLCanvasElement>("#test-canvas")!;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
if (simMode) {
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
}

const scene = new THREE.Scene();
scene.background = new THREE.Color("#000000");
scene.add(new THREE.AmbientLight(0xffffff, simMode ? 0.35 : 0.6));
const sun = new THREE.DirectionalLight(0xffffff, simMode ? 3.0 : 2.5);
if (simMode) {
  sun.position.set(-30, -20, -15);
} else {
  sun.position.set(3, 5, 4);
}
scene.add(sun);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.01, 100);
if (simMode) {
  camera.position.set(-12, 12, 8);
} else {
  camera.position.set(2.2, 1.4, 2.2);
}
camera.lookAt(0, 0, 0);

const loader = new FBXLoader();
loader.load(
  `/models/${modelName}.fbx`,
  (model) => {
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    model.scale.setScalar(2 / maxDim);
    const center = new THREE.Box3().setFromObject(model).getCenter(new THREE.Vector3());
    model.position.sub(center);
    scene.add(model);

    const dumpMaterials = (label: string) => {
      log(`--- ${label} ---`);
      model.traverse((child) => {
        const mesh = child as THREE.Mesh;
        if (!mesh.isMesh) return;
        const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        const hasUV = Boolean((mesh.geometry as THREE.BufferGeometry).attributes.uv);
        for (const mat of materials) {
          const m = mat as THREE.MeshPhongMaterial;
          log(
            `mesh=${mesh.name} uv=${hasUV} mat=${m.name} type=${m.type} ` +
            `map=${m.map ? (m.map.image ? `${m.map.image.width}x${m.map.image.height}` : "no-image") : "null"} ` +
            `color=#${m.color?.getHexString?.() ?? "?"}`,
          );
        }
      });
    };

    log(`model: ${modelName}`);
    dumpMaterials("on load");
    // Textures decode asynchronously; re-check the final state after 3 seconds
    setTimeout(() => {
      dumpMaterials("after 3s");
      log("DONE");
    }, 3000);
  },
  undefined,
  (err) => log(`LOAD ERROR: ${String(err)}`),
);

function animate() {
  requestAnimationFrame(animate);
  scene.rotation.y += 0.004;
  renderer.render(scene, camera);
}
animate();
