// Usage:
//   cmd /c npm install
//   cmd /c npm run dev:bridge
// Overview:
//   The authoritative simulation core of the environment, aligned with the
//   UE/AirSim fly_redis.py behavior:
//   - No orbital dynamics; pose commands are displacements (dx/dy/dz applied directly, no drift)
//   - The target satellite can spin at a fixed rate (E1 spinning-target scenario)
//   - LiDAR point clouds are generated from FBX surface samples sent back by the
//     browser, expressed in the camera frame with field-of-view culling
//   - Supports degradation injection: actuation noise E3N, thruster fault E3F,
//     intermittent LiDAR dropout E4L

import fs from "node:fs";
import path from "node:path";
import type {
  AnnotationRoot,
  CaptureMetrics,
  DeltaPoseCommand,
  EventRecord,
  HiddenTruthSnapshot,
  LidarMessage,
  PoseTruthMessage,
  SceneConfig,
  ServiceSite,
  SatelliteAnnotations,
  Vec3,
} from "../shared/protocol.ts";
import {
  add,
  addEuler,
  angleBetween,
  bodyAxesInWorld,
  bodyToWorld,
  computePoseTruth,
  norm,
  randomGaussian,
  roundNumber,
  scale,
  sub,
  timestampNs,
  vec3,
  withinRange,
  worldToBody,
} from "../shared/math.ts";

type SimState = {
  sim_time_s: number;
  servicer_position_world_m: Vec3;
  servicer_attitude_rpy_rad: { pitch: number; roll: number; yaw: number };
  target_position_world_m: Vec3;
  target_attitude_rpy_rad: { pitch: number; roll: number; yaw: number };
  exposure_value: number;
  collision_count: number;
};

export class RelativeOrbitSim {
  public readonly sceneConfig: SceneConfig;
  public readonly targetAnnotations: SatelliteAnnotations;
  private readonly eventLogPath: string;
  private readonly hiddenTruthLogPath: string;
  private readonly servicerRadiusM: number;
  private readonly viewpointHoldCounters = new Map<string, number>();
  // Model surface samples in the target body frame, sent back by the browser after
  // the FBX loads; approximated by a sphere until then
  private targetSurfacePointsBody: Vec3[];
  private targetRadiusM: number;
  private state: SimState;
  private latestMetrics: CaptureMetrics | null = null;
  private lastVisibilityRatio = 0;

  constructor(sceneConfig: SceneConfig, annotations: AnnotationRoot, logDir: string) {
    this.sceneConfig = sceneConfig;
    const targetName = sceneConfig.target.name;
    const targetAnnotations = annotations.satellites?.[targetName];
    if (!targetAnnotations) {
      throw new Error(`Missing annotations for target ${targetName}`);
    }
    this.targetAnnotations = targetAnnotations;

    const modelConfig = sceneConfig.satellites?.[targetName];
    if (!modelConfig) {
      throw new Error(`Missing satellite model config for target ${targetName}`);
    }

    const sessionStamp = new Date().toISOString().replace(/[:.]/g, "-");
    this.eventLogPath = path.join(logDir, `threejs_env_events_${sessionStamp}.jsonl`);
    this.hiddenTruthLogPath = path.join(logDir, `threejs_env_truth_${sessionStamp}.jsonl`);

    this.state = {
      sim_time_s: 0,
      servicer_position_world_m: { ...sceneConfig.servicer.initial_state.position_world_m },
      servicer_attitude_rpy_rad: { ...sceneConfig.servicer.initial_state.attitude_rpy_rad },
      target_position_world_m: { x: 0, y: 0, z: 0 },
      target_attitude_rpy_rad: { pitch: 0, roll: 0, yaw: 0 },
      exposure_value: 0,
      collision_count: 0,
    };

    this.targetRadiusM = modelConfig.max_diameter_m / 2;
    this.targetSurfacePointsBody = buildSphereSurfacePoints(this.targetRadiusM);
    this.servicerRadiusM = this.boxRadius(sceneConfig.servicer.body_size_m);
  }

  // No dynamics: only advance time and target spin; the servicer pose is fully command-driven
  public step(dt: number): void {
    const spinRad = (this.sceneConfig.target.spin_deg_s * Math.PI) / 180;
    if (spinRad !== 0) {
      this.state.target_attitude_rpy_rad = addEuler(this.state.target_attitude_rpy_rad, {
        pitch: 0,
        roll: 0,
        yaw: spinRad * dt,
      });
    }
    this.state.sim_time_s += dt;
  }

  public setTargetModelInfo(surfacePointsBody: number[], boundingRadiusM: number): void {
    const points: Vec3[] = [];
    for (let i = 0; i + 2 < surfacePointsBody.length; i += 3) {
      points.push(vec3(surfacePointsBody[i], surfacePointsBody[i + 1], surfacePointsBody[i + 2]));
    }
    if (points.length > 0) {
      this.targetSurfacePointsBody = points;
      this.targetRadiusM = boundingRadiusM;
      this.recordEvent("target_model_loaded", {
        surface_point_count: points.length,
        bounding_radius_m: roundNumber(boundingRadiusM),
      });
    }
  }

  public applyAbsolutePose(command: Record<string, unknown>): void {
    this.state.servicer_position_world_m = {
      x: Number(command.x ?? this.state.servicer_position_world_m.x),
      y: Number(command.y ?? this.state.servicer_position_world_m.y),
      z: Number(command.z ?? this.state.servicer_position_world_m.z),
    };
    this.state.servicer_attitude_rpy_rad = {
      pitch: Number(command.pitch ?? this.state.servicer_attitude_rpy_rad.pitch),
      roll: Number(command.roll ?? this.state.servicer_attitude_rpy_rad.roll),
      yaw: Number(command.yaw ?? this.state.servicer_attitude_rpy_rad.yaw),
    };
    this.recordEvent("pose_applied", { mode: "absolute", command });
    this.updateCollisionState();
  }

  // Command equals displacement: body-frame deltas are rotated to the world frame
  // and applied directly, with no residual velocity
  public applyDeltaPose(command: DeltaPoseCommand): void {
    const injected = this.injectFault(this.injectNoise(command));
    const deltaBody = vec3(injected.dx ?? 0, injected.dy ?? 0, injected.dz ?? 0);
    const deltaWorld = bodyToWorld(deltaBody, this.state.servicer_attitude_rpy_rad);
    this.state.servicer_position_world_m = add(this.state.servicer_position_world_m, deltaWorld);

    this.state.servicer_attitude_rpy_rad = addEuler(this.state.servicer_attitude_rpy_rad, {
      pitch: injected.dpitch ?? 0,
      roll: injected.droll ?? 0,
      yaw: injected.dyaw ?? 0,
    });

    this.recordEvent("pose_applied", {
      mode: "delta",
      command,
      executed_delta_body_m: deltaBody,
      delta_world_m: deltaWorld,
    });
    this.updateCollisionState();
  }

  public setExposure(value: number): void {
    this.state.exposure_value = Math.max(-3, Math.min(3, value));
    this.recordEvent("exposure_changed", { exposure_value: this.state.exposure_value });
  }

  public buildRenderState() {
    const modelConfig = this.sceneConfig.satellites[this.sceneConfig.target.name];
    return {
      timestamp: timestampNs(),
      scene_name: this.sceneConfig.scene_name,
      exposure_value: this.state.exposure_value,
      target_name: this.sceneConfig.target.name,
      camera_offset_body_m: this.sceneConfig.camera_offset_body_m,
      servicer: {
        position_world_m: { ...this.state.servicer_position_world_m },
        attitude_rpy_rad: { ...this.state.servicer_attitude_rpy_rad },
      },
      target: {
        position_world_m: { ...this.state.target_position_world_m },
        attitude_rpy_rad: { ...this.state.target_attitude_rpy_rad },
        model_url: modelConfig.model_url,
        max_diameter_m: modelConfig.max_diameter_m,
      },
    };
  }

  public buildPoseTruthMessage(): PoseTruthMessage {
    const pose = computePoseTruth({
      servicePosition: this.state.servicer_position_world_m,
      serviceAttitude: this.state.servicer_attitude_rpy_rad,
      serviceVelocity: vec3(0, 0, 0),
      serviceAngularVelocity: { pitch: 0, roll: 0, yaw: 0 },
      targetPosition: this.state.target_position_world_m,
      cameraOffsetBody: this.sceneConfig.camera_offset_body_m,
    });
    return { ...pose, timestamp: timestampNs() };
  }

  // Camera-frame point cloud: model surface points -> target attitude rotation ->
  // world frame -> camera frame, with 60°x45° field-of-view culling
  public buildLidarMessage(): LidarMessage {
    const dropout = this.sceneConfig.injection.lidar_dropout;
    if (dropout > 0 && Math.random() < dropout) {
      this.recordEvent("lidar_dropout_injected", { probability: dropout });
      return { timestamp: timestampNs(), points: [], total_points: 0 };
    }

    const pointsCamera: number[] = [];
    const serviceAttitude = this.state.servicer_attitude_rpy_rad;
    const servicePosition = this.state.servicer_position_world_m;
    const cameraOffsetWorld = bodyToWorld(this.sceneConfig.camera_offset_body_m, serviceAttitude);
    const cameraWorld = add(servicePosition, cameraOffsetWorld);

    for (const pointBodyTarget of this.targetSurfacePointsBody) {
      const pointWorld = add(
        this.state.target_position_world_m,
        bodyToWorld(pointBodyTarget, this.state.target_attitude_rpy_rad),
      );
      const fromCamera = sub(pointWorld, cameraWorld);
      const pointBody = worldToBody(fromCamera, serviceAttitude);
      const azimuth = Math.atan2(pointBody.y, pointBody.x || 1e-9);
      const elevation = Math.atan2(-pointBody.z, Math.sqrt(pointBody.x ** 2 + pointBody.y ** 2) || 1e-9);
      if (pointBody.x <= 0) continue;
      if (Math.abs(azimuth) > Math.PI / 3) continue;
      if (Math.abs(elevation) > Math.PI / 4) continue;
      pointsCamera.push(roundNumber(pointBody.x), roundNumber(pointBody.y), roundNumber(pointBody.z));
    }

    const limited = pointsCamera.length > 900 ? pointsCamera.slice(0, 900) : pointsCamera;
    return {
      timestamp: timestampNs(),
      points: limited,
      total_points: Math.floor(limited.length / 3),
    };
  }

  public setLatestMetrics(metrics: CaptureMetrics | null): void {
    this.latestMetrics = metrics;
    const visibility = metrics?.visibility_ratio ?? 0;
    if (this.lastVisibilityRatio >= 0.15 && visibility < 0.15) {
      this.recordEvent("visibility_drop", {
        previous_visibility_ratio: this.lastVisibilityRatio,
        visibility_ratio: visibility,
      });
    }
    this.lastVisibilityRatio = visibility;
  }

  public buildHiddenTruthSnapshot(): HiddenTruthSnapshot {
    const poseTruth = this.buildPoseTruthMessage();
    const viewpointHits = this.computeViewpointHits(poseTruth, this.latestMetrics);
    const serviceSiteHits = this.computeServiceSiteHits(poseTruth);
    const snapshot: HiddenTruthSnapshot = {
      timestamp: timestampNs(),
      scene_name: this.sceneConfig.scene_name,
      target_name: this.sceneConfig.target.name,
      sim_time_s: roundNumber(this.state.sim_time_s, 3),
      exposure_value: roundNumber(this.state.exposure_value, 3),
      collision_count: this.state.collision_count,
      pose_truth: poseTruth,
      render_metrics: this.latestMetrics,
      viewpoint_hits: viewpointHits,
      service_site_hits: serviceSiteHits,
    };

    this.appendJsonLine(this.hiddenTruthLogPath, snapshot);
    return snapshot;
  }

  public recordEvent(type: string, payload: Record<string, unknown>): void {
    const record: EventRecord = { timestamp: timestampNs(), type, payload };
    this.appendJsonLine(this.eventLogPath, record);
  }

  private updateCollisionState(): void {
    const distance = norm(sub(this.state.servicer_position_world_m, this.state.target_position_world_m));
    if (distance <= this.servicerRadiusM + this.targetRadiusM) {
      this.state.collision_count += 1;
      this.recordEvent("collision", {
        collision_count: this.state.collision_count,
        distance_m: roundNumber(distance),
      });
    }
  }

  // E3N actuation noise: add Gaussian noise to command deltas
  private injectNoise(command: DeltaPoseCommand): DeltaPoseCommand {
    if (!this.sceneConfig.noise.enabled) {
      return command;
    }
    return {
      ...command,
      dx: Number(command.dx ?? 0) + randomGaussian(this.sceneConfig.noise.position_sigma_m),
      dy: Number(command.dy ?? 0) + randomGaussian(this.sceneConfig.noise.position_sigma_m),
      dz: Number(command.dz ?? 0) + randomGaussian(this.sceneConfig.noise.position_sigma_m),
      dpitch: Number(command.dpitch ?? 0) + randomGaussian(this.sceneConfig.noise.attitude_sigma_rad),
      droll: Number(command.droll ?? 0) + randomGaussian(this.sceneConfig.noise.attitude_sigma_rad),
      dyaw: Number(command.dyaw ?? 0) + randomGaussian(this.sceneConfig.noise.attitude_sigma_rad),
    };
  }

  // E3F thruster fault: translation on the specified axis only executes at fault_scale
  private injectFault(command: DeltaPoseCommand): DeltaPoseCommand {
    const axis = this.sceneConfig.injection.fault_axis;
    if (!axis) {
      return command;
    }
    const scaled = { ...command };
    scaled[axis] = Number(command[axis] ?? 0) * this.sceneConfig.injection.fault_scale;
    return scaled;
  }

  private computeViewpointHits(poseTruth: PoseTruthMessage, metrics: CaptureMetrics | null): string[] {
    const hits: string[] = [];
    const rel = poseTruth.relative_position;
    const visibilityRatio = metrics?.visibility_ratio ?? 0;

    for (const viewpoint of this.targetAnnotations.survey_viewpoints) {
      const region = viewpoint.target_region_body_frame_m;
      const inRegion =
        withinRange(rel.x, region.x) &&
        withinRange(rel.y, region.y) &&
        withinRange(rel.z, region.z) &&
        visibilityRatio >= viewpoint.visibility_ratio_min;
      const currentCount = this.viewpointHoldCounters.get(viewpoint.id) ?? 0;
      const nextCount = inRegion ? currentCount + 1 : 0;
      this.viewpointHoldCounters.set(viewpoint.id, nextCount);
      if (nextCount >= viewpoint.hold_steps_min) {
        hits.push(viewpoint.id);
      }
    }

    if (hits.length > 0) {
      this.recordEvent("viewpoint_hit", { viewpoint_ids: hits });
    }
    return hits;
  }

  private computeServiceSiteHits(poseTruth: PoseTruthMessage): string[] {
    const hits: string[] = [];
    const rel = poseTruth.relative_position;
    const axes = bodyAxesInWorld(this.state.servicer_attitude_rpy_rad);
    const desiredForward = sub(this.state.target_position_world_m, poseTruth.camera_pose.position);

    for (const site of this.targetAnnotations.service_sites) {
      if (this.isServiceSiteAligned(site, rel, axes.forward, desiredForward, poseTruth.distance)) {
        hits.push(site.id);
      }
    }

    if (hits.length > 0) {
      this.recordEvent("service_site_alignment_hit", { service_site_ids: hits });
    }
    return hits;
  }

  private isServiceSiteAligned(
    site: ServiceSite,
    relativePositionBody: Vec3,
    servicerForwardWorld: Vec3,
    desiredForwardWorld: Vec3,
    distance: number,
  ): boolean {
    const error = norm(sub(relativePositionBody, site.position_body_m));
    if (error > site.position_tolerance_m) return false;
    if (!withinRange(distance, site.stand_off_distance_range_m)) return false;

    const attitudeErrorDeg = roundNumber((angleBetween(servicerForwardWorld, desiredForwardWorld) * 180) / Math.PI, 2);
    if (attitudeErrorDeg > site.attitude_tolerance_deg) return false;

    const siteApproachAxisWorld = bodyToWorld(site.preferred_approach_axis_body, this.state.target_attitude_rpy_rad);
    const currentApproachWorld = normalizeSafe(sub(this.state.servicer_position_world_m, this.state.target_position_world_m));
    const approachErrorDeg = roundNumber((angleBetween(siteApproachAxisWorld, currentApproachWorld) * 180) / Math.PI, 2);
    return approachErrorDeg <= site.approach_axis_tolerance_deg;
  }

  private boxRadius(size: { x: number; y: number; z: number }): number {
    return Math.sqrt(size.x ** 2 + size.y ** 2 + size.z ** 2) / 2;
  }

  private appendJsonLine(filePath: string, payload: unknown): void {
    fs.appendFileSync(filePath, `${JSON.stringify(payload)}\n`, "utf-8");
  }
}

// Placeholder before the FBX finishes loading: uniform samples on the bounding sphere
function buildSphereSurfacePoints(radius: number, count = 300): Vec3[] {
  const points: Vec3[] = [];
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i += 1) {
    const y = 1 - (i / (count - 1)) * 2;
    const r = Math.sqrt(1 - y * y);
    const theta = goldenAngle * i;
    points.push(vec3(radius * r * Math.cos(theta), radius * y, radius * r * Math.sin(theta)));
  }
  return points;
}

function normalizeSafe(v: Vec3): Vec3 {
  const length = norm(v);
  if (length <= 1e-9) {
    return vec3(1, 0, 0);
  }
  return scale(v, 1 / length);
}
