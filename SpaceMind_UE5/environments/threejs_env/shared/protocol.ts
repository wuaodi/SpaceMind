export type Vec3 = { x: number; y: number; z: number };
export type Euler = { pitch: number; roll: number; yaw: number };

export type AbsolutePoseCommand = {
  x?: number;
  y?: number;
  z?: number;
  pitch?: number;
  roll?: number;
  yaw?: number;
  timestamp?: string;
};

export type DeltaPoseCommand = {
  dx?: number;
  dy?: number;
  dz?: number;
  dpitch?: number;
  droll?: number;
  dyaw?: number;
  timestamp?: string;
};

export type ExposureCommand = {
  exposure_value: number;
};

export type PrimitiveBox = {
  x: number;
  y: number;
  z: number;
};

export type SatelliteModelConfig = {
  model_url: string;
  max_diameter_m: number;
};

// 退化注入开关，与 fly_redis.py 的 CLI 一一对应（E3N/E3F/E4L/E4E）
export type InjectionConfig = {
  fault_axis: "" | "dx" | "dy" | "dz";
  fault_scale: number;
  lidar_dropout: number;
  exposure_disturb_step: number;
  exposure_disturb_value: number;
};

export type SceneConfig = {
  scene_name: string;
  sim_hz: number;
  publish_hz: number;
  capture_width: number;
  capture_height: number;
  camera_offset_body_m: Vec3;
  noise: {
    enabled: boolean;
    position_sigma_m: number;
    attitude_sigma_rad: number;
  };
  injection: InjectionConfig;
  servicer: {
    body_size_m: PrimitiveBox;
    initial_state: {
      position_world_m: Vec3;
      attitude_rpy_rad: Euler;
    };
  };
  target: {
    name: string;
    spin_deg_s: number;
  };
  satellites: Record<string, SatelliteModelConfig>;
};

export type SurveyViewpoint = {
  id: string;
  description: string;
  target_region_body_frame_m: {
    x: [number, number];
    y: [number, number];
    z: [number, number];
  };
  hold_steps_min: number;
  visibility_ratio_min: number;
};

export type ServiceSite = {
  id: string;
  description: string;
  position_body_m: Vec3;
  position_tolerance_m: number;
  stand_off_distance_range_m: [number, number];
  preferred_approach_axis_body: Vec3;
  approach_axis_tolerance_deg: number;
  attitude_tolerance_deg: number;
};

export type SatelliteAnnotations = {
  inspection_labels: {
    type: string;
    status: string;
    function: string;
    key_components: string[];
  };
  survey_viewpoints: SurveyViewpoint[];
  service_sites: ServiceSite[];
};

export type AnnotationRoot = {
  satellites: Record<string, SatelliteAnnotations>;
};

export type RenderState = {
  timestamp: string;
  scene_name: string;
  exposure_value: number;
  target_name: string;
  camera_offset_body_m: Vec3;
  servicer: {
    position_world_m: Vec3;
    attitude_rpy_rad: Euler;
  };
  target: {
    position_world_m: Vec3;
    attitude_rpy_rad: Euler;
    model_url: string;
    max_diameter_m: number;
  };
};

export type CaptureRequestMessage = {
  type: "capture_request";
  capture_id: string;
  width: number;
  height: number;
  reason: string;
  render_state: RenderState;
};

export type CaptureMetrics = {
  target_visible: boolean;
  visibility_ratio: number;
  projected_bounds_px: {
    min_x: number;
    min_y: number;
    max_x: number;
    max_y: number;
  } | null;
};

export type CaptureResponseMessage = {
  type: "capture_response";
  capture_id: string;
  width: number;
  height: number;
  rgb_png_base64: string;
  segmentation_png_base64?: string;
  metrics: CaptureMetrics;
};

export type ReadyMessage = {
  type: "ready";
};

// 浏览器加载 FBX 后回传：目标体坐标系下的表面采样点（展平的 xyz 序列）与包围半径
export type ModelInfoMessage = {
  type: "model_info";
  target_name: string;
  bounding_radius_m: number;
  surface_points_body: number[];
  part_names: string[];
};

export type SceneInitMessage = {
  type: "scene_init";
  scene_config: SceneConfig;
};

export type StateSnapshotMessage = {
  type: "state_snapshot";
  render_state: RenderState;
};

export type ErrorMessage = {
  type: "error";
  message: string;
};

export type BridgeToAppMessage = SceneInitMessage | StateSnapshotMessage | CaptureRequestMessage | ErrorMessage;
export type AppToBridgeMessage = ReadyMessage | CaptureResponseMessage | ModelInfoMessage | ErrorMessage;

export type ImageMessage = {
  name: string;
  timestamp: string;
  width: number;
  height: number;
  data: string;
};

export type SegmentationMessage = ImageMessage;

export type LidarMessage = {
  timestamp: string;
  points: number[];
  total_points: number;
};

export type PoseTruthMessage = {
  relative_position: Vec3;
  distance: number;
  azimuth_deg: number;
  elevation_deg: number;
  service_spacecraft_pose: {
    position: Vec3;
    attitude_rpy_rad: Euler;
    velocity_world_mps: Vec3;
    angular_velocity_rpy_rad_s: Euler;
  };
  camera_pose: {
    position: Vec3;
    forward_world: Vec3;
    up_world: Vec3;
  };
  timestamp: string;
};

export type HiddenTruthSnapshot = {
  timestamp: string;
  scene_name: string;
  target_name: string;
  sim_time_s: number;
  exposure_value: number;
  collision_count: number;
  pose_truth: PoseTruthMessage;
  render_metrics: CaptureMetrics | null;
  viewpoint_hits: string[];
  service_site_hits: string[];
};

export type EventRecord = {
  timestamp: string;
  type: string;
  payload: Record<string, unknown>;
};
