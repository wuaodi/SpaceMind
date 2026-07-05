import type { Euler, PoseTruthMessage, Vec3 } from "./protocol.ts";

export function vec3(x = 0, y = 0, z = 0): Vec3 {
  return { x, y, z };
}

export function add(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}

export function sub(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

export function scale(v: Vec3, s: number): Vec3 {
  return { x: v.x * s, y: v.y * s, z: v.z * s };
}

export function dot(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

export function norm(v: Vec3): number {
  return Math.sqrt(dot(v, v));
}

export function normalize(v: Vec3): Vec3 {
  const n = norm(v);
  if (n <= 1e-9) return vec3(0, 0, 0);
  return scale(v, 1 / n);
}

export function clamp(value: number, minValue: number, maxValue: number): number {
  return Math.min(maxValue, Math.max(minValue, value));
}

export function deg(valueRad: number): number {
  return (valueRad * 180) / Math.PI;
}

export function normalizeAngle(angle: number): number {
  let out = angle;
  while (out > Math.PI) out -= Math.PI * 2;
  while (out < -Math.PI) out += Math.PI * 2;
  return out;
}

export function addEuler(a: Euler, b: Euler): Euler {
  return {
    pitch: normalizeAngle(a.pitch + b.pitch),
    roll: normalizeAngle(a.roll + b.roll),
    yaw: normalizeAngle(a.yaw + b.yaw),
  };
}

export function scaleEuler(a: Euler, s: number): Euler {
  return { pitch: a.pitch * s, roll: a.roll * s, yaw: a.yaw * s };
}

export function rotationMatrixFromEuler(euler: Euler): number[][] {
  const cp = Math.cos(euler.pitch);
  const sp = Math.sin(euler.pitch);
  const cr = Math.cos(euler.roll);
  const sr = Math.sin(euler.roll);
  const cy = Math.cos(euler.yaw);
  const sy = Math.sin(euler.yaw);

  const rz = [
    [cy, -sy, 0],
    [sy, cy, 0],
    [0, 0, 1],
  ];
  const ry = [
    [cp, 0, sp],
    [0, 1, 0],
    [-sp, 0, cp],
  ];
  const rx = [
    [1, 0, 0],
    [0, cr, -sr],
    [0, sr, cr],
  ];

  return multiplyMatrix(multiplyMatrix(rz, ry), rx);
}

function multiplyMatrix(a: number[][], b: number[][]): number[][] {
  const out = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let r = 0; r < 3; r += 1) {
    for (let c = 0; c < 3; c += 1) {
      out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
    }
  }
  return out;
}

export function transposeMatrix(m: number[][]): number[][] {
  return [
    [m[0][0], m[1][0], m[2][0]],
    [m[0][1], m[1][1], m[2][1]],
    [m[0][2], m[1][2], m[2][2]],
  ];
}

export function transformVec(m: number[][], v: Vec3): Vec3 {
  return {
    x: m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
    y: m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
    z: m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
  };
}

export function bodyToWorld(v: Vec3, euler: Euler): Vec3 {
  return transformVec(rotationMatrixFromEuler(euler), v);
}

export function worldToBody(v: Vec3, euler: Euler): Vec3 {
  return transformVec(transposeMatrix(rotationMatrixFromEuler(euler)), v);
}

export function bodyAxesInWorld(euler: Euler): { forward: Vec3; right: Vec3; down: Vec3; up: Vec3 } {
  const forward = normalize(bodyToWorld(vec3(1, 0, 0), euler));
  const right = normalize(bodyToWorld(vec3(0, 1, 0), euler));
  const down = normalize(bodyToWorld(vec3(0, 0, 1), euler));
  const up = scale(down, -1);
  return { forward, right, down, up };
}

export function timestampNs(): string {
  return `${BigInt(Date.now()) * 1000000n}`;
}

export function randomGaussian(stddev: number): number {
  if (stddev <= 0) return 0;
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return stddev * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export function computeCameraPose(servicePosition: Vec3, attitude: Euler, cameraOffsetBody: Vec3) {
  const axes = bodyAxesInWorld(attitude);
  const offsetWorld = bodyToWorld(cameraOffsetBody, attitude);
  return {
    position: add(servicePosition, offsetWorld),
    forward_world: axes.forward,
    up_world: axes.up,
  };
}

export function computePoseTruth(args: {
  servicePosition: Vec3;
  serviceAttitude: Euler;
  serviceVelocity: Vec3;
  serviceAngularVelocity: Euler;
  targetPosition?: Vec3;
  cameraOffsetBody?: Vec3;
}): PoseTruthMessage {
  const targetPosition = args.targetPosition ?? vec3(0, 0, 0);
  const cameraOffsetBody = args.cameraOffsetBody ?? vec3(1, 0, 0);
  const cameraPose = computeCameraPose(args.servicePosition, args.serviceAttitude, cameraOffsetBody);
  const relativeWorld = sub(targetPosition, args.servicePosition);
  const relativeBody = worldToBody(relativeWorld, args.serviceAttitude);
  const relativeCamera = {
    x: relativeBody.x - cameraOffsetBody.x,
    y: relativeBody.y - cameraOffsetBody.y,
    z: relativeBody.z - cameraOffsetBody.z,
  };
  const distance = norm(relativeCamera);
  const azimuth = Math.atan2(relativeCamera.y, relativeCamera.x || 1e-9);
  const elevation = Math.atan2(-relativeCamera.z, Math.sqrt(relativeCamera.x ** 2 + relativeCamera.y ** 2) || 1e-9);
  return {
    relative_position: roundVec(relativeCamera),
    distance: roundNumber(distance),
    azimuth_deg: roundNumber(deg(azimuth), 1),
    elevation_deg: roundNumber(deg(elevation), 1),
    service_spacecraft_pose: {
      position: roundVec(args.servicePosition),
      attitude_rpy_rad: roundEuler(args.serviceAttitude),
      velocity_world_mps: roundVec(args.serviceVelocity),
      angular_velocity_rpy_rad_s: roundEuler(args.serviceAngularVelocity),
    },
    camera_pose: {
      position: roundVec(cameraPose.position),
      forward_world: roundVec(cameraPose.forward_world),
      up_world: roundVec(cameraPose.up_world),
    },
    timestamp: timestampNs(),
  };
}

export function roundNumber(value: number, digits = 3): number {
  const scaleValue = 10 ** digits;
  return Math.round(value * scaleValue) / scaleValue;
}

export function roundVec(v: Vec3, digits = 3): Vec3 {
  return {
    x: roundNumber(v.x, digits),
    y: roundNumber(v.y, digits),
    z: roundNumber(v.z, digits),
  };
}

export function roundEuler(euler: Euler, digits = 4): Euler {
  return {
    pitch: roundNumber(euler.pitch, digits),
    roll: roundNumber(euler.roll, digits),
    yaw: roundNumber(euler.yaw, digits),
  };
}

export function withinRange(value: number, range: [number, number]): boolean {
  return value >= range[0] && value <= range[1];
}

export function angleBetween(a: Vec3, b: Vec3): number {
  const na = normalize(a);
  const nb = normalize(b);
  const cosine = clamp(dot(na, nb), -1, 1);
  return Math.acos(cosine);
}
