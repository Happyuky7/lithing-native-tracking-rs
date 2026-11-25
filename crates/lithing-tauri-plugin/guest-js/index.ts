/**
 * Web-side client of the native tracker (Tauri plugin `lithing-tracking`).
 *
 * `NativeFaceTracker` has the public surface main.ts uses on the web
 * `FaceTracker` (src/app/face-tracker.ts): initialize/start/stop/update/predict,
 * getTrackingData/getHands/getBlendshapes and every set* option (low / high
 * light, glasses, mirror, stability, rates, hand quality, finger smoothing,
 * tongue, delegate; the worker options are accepted and ignored) plus the
 * native-only camera options `setKeepFrameRate` (default on) and
 * `setCameraMode` (default Auto; `listCameraFormats` lists the modes), the cheek
 * puff tuning `setCheekTuning` and the calibration captures `captureLevel`
 * (tongue in / out, cheeks relaxed / puffed), so the app
 * can pick one or the other behind a setting. Detection runs in Rust on its
 * own threads; this class only decodes the ~600-byte frame each result
 * arrives as (no model, no camera, no canvas on the render thread).
 */
import { Channel, invoke } from '@tauri-apps/api/core';
import { decodeFrame, TRACKING_KEYS, type HandPose, type NativeFrame, type TrackingData } from './decode';

export * from './decode';

export type HandTrackingQuality = 'performance' | 'balanced' | 'precise';
export interface TongueTuning { sensitivity: number; inLevel: number; outLevel: number }
/** Mirrors lithing_tracking::cheek::CheekTuning: raw cheek widths captured relaxed / puffed (0: not captured). */
export interface CheekTuning { sensitivity: number; relaxedLevel: number; puffedLevel: number }
/** What captureLevel() averages: the tongue detector's extension or the cheek detector's width. */
export type CaptureKind = 'tongue' | 'cheeks';
/** Mirrors lithing_tracking::eyes::EyeCalibration (and EyeCalibration in the app's eye-openness.ts). */
export interface EyeCalibration {
  right: Record<'blink' | 'gap' | 'iris', { open: number; closed: number }>;
  left: Record<'blink' | 'gap' | 'iris', { open: number; closed: number }>;
  capturedAt: number;
}

/** Mirrors lithing_tracking::gaze::GazeCalibration (and GazeCalibration in the app's gaze.ts): raw gaze at the centre and each screen edge. */
export interface GazeCalibration {
  center: { x: number; y: number };
  left: number; right: number; up: number; down: number;
  capturedAt: number;
}

/** Mirrors lithing_tracking::TrackerSettings (serde camelCase). */
export interface NativeTrackerSettings {
  faceFps: number;
  handFps: number;
  handsEnabled: boolean;
  tongueEnabled: boolean;
  tongueTuning: TongueTuning;
  tongueFps: number;
  cheekTuning: CheekTuning;
  face: { mirror: boolean; glassesMode: boolean; lowLight: boolean; highLight: boolean; stability: number; eyeOpenness: number; eyeCalibration: EyeCalibration | null; gazeCalibration: GazeCalibration | null };
  hands: { mirror: boolean; quality: HandTrackingQuality; fingerSmoothing: number; lowLight: boolean; highLight: boolean };
  includeRaw: boolean;
}

/** One camera mode (lithing_tracking::source::CameraMode). `format` absent = any format of that size and rate. */
export interface NativeCameraMode {
  format?: 'MJPEG' | 'YUYV' | 'NV12' | 'GRAY' | 'RAWRGB' | 'RAWBGR' | (string & {}) | null;
  width: number;
  height: number;
  fps: number;
}

export interface NativeStartOptions {
  /** `keepFrameRate` (default true): keep the camera's rate in low light; restored on stop. */
  camera?: { index: number; width: number; height: number; fps: number; keepFrameRate?: boolean };
  /** Explicit camera mode; null / absent = Auto. A mode the camera rejects opens in Auto (status().cameraWarning). */
  cameraMode?: NativeCameraMode | null;
  videoFile?: string;
  provider?: 'auto' | 'directml' | 'cuda' | 'cpu' | 'nnapi';
  cpuThreads?: number;
  priority?: 'lowest' | 'below-normal' | 'normal' | 'above-normal';
  palmInterval?: number;
  modelsDir?: string;
  wire?: { handLandmarks?: boolean; faceLandmarks?: boolean };
}

export type CameraControlKind = 'autoExposurePriority' | 'variableFrameRate' | 'exposure' | 'gain' | 'brightness' | 'backlightCompensation';

/** One camera control as the driver describes it (exposure: log2 seconds on Windows, 100 us on Linux). */
export interface NativeControlInfo {
  kind: CameraControlKind; name: string; min: number; max: number; step: number; default: number; value: number;
  auto: boolean; supportsAuto: boolean; supportsManual: boolean;
}

export interface NativeControlChange {
  kind: CameraControlKind; from: { value: number; auto: boolean } | null; to: { value: number; auto: boolean };
  reason: string; error: string | null; at: number;
}

/** Mirrors lithing_camera_controls::CameraControlReport: what "Keep frame rate" found and did. */
export interface NativeCameraControlReport {
  device: string;
  keepFrameRate: boolean;
  /** NVIDIA Broadcast, OBS Virtual Camera, AMD Privacy View...: nothing is changed. */
  virtualCamera: boolean;
  controls: NativeControlInfo[];
  controlsError: string | null;
  /** 'none' | 'priority' | 'exposure-cap-on-drop' | 'exposure-cap' | 'unsupported' | 'off' */
  strategy: string;
  changes: NativeControlChange[];
  restored: boolean;
  restoreErrors: string[];
  targetFps: number;
  fpsBefore: number | null;
  fpsAfter: number | null;
  luma: number | null;
  notes: string[];
}

export interface NativeDeviceControls {
  device: { index: number; name: string; id: string; virtualCamera: boolean };
  controls: NativeControlInfo[];
  /** Why the device was not opened (virtual, busy in another app) or could not be read. */
  skipped: string | null;
}

export interface NativeStatus {
  running: boolean; framesSent: number; width: number; height: number; fps: number; source: string;
  faceProvider: string; handProvider: string; loadMs: number; dropped: number; lastError?: string;
  /** The camera's "Keep frame rate" report (absent for video files and older plugins). */
  cameraControls?: NativeCameraControlReport | null;
  /** Pixel format the camera delivers: 'MJPEG', 'YUYV', 'NV12'... (absent in older plugins). */
  format?: string;
  /** The camera mode asked for at start (null = Auto). */
  cameraMode?: NativeCameraMode | null;
  /** Why the camera did not open in the requested mode (it opened in Auto instead). */
  cameraWarning?: string | null;
}

const CAPTURE_SIZE: Record<HandTrackingQuality, { width: number; height: number }> = {
  performance: { width: 640, height: 480 }, balanced: { width: 640, height: 480 }, precise: { width: 1280, height: 720 }
};

const command = (name: string) => `plugin:lithing-tracking|${name}`;

/**
 * Port of src/app/motion-predictor.ts: carries the last detection on at its
 * velocity for up to one detection interval (dead reckoning), for render
 * frames between detections.
 */
class MotionPredictor {
  readonly value: Float64Array;
  readonly velocity: Float64Array;
  private time = -Infinity;
  private interval = 1000 / 30;
  readonly size: number;
  constructor(size: number) {
    this.size = size;
    this.value = new Float64Array(size);
    this.velocity = new Float64Array(size);
  }
  reset(): void { this.time = -Infinity; this.velocity.fill(0); }
  push(values: ArrayLike<number>, time: number): void {
    const elapsed = time - this.time;
    const continuous = elapsed > 0.5 && elapsed < 160;
    if (continuous) this.interval += (Math.min(70, elapsed) - this.interval) * 0.15;
    for (let index = 0; index < this.size; index++) {
      const next = values[index];
      if (!continuous) { this.velocity[index] = 0; this.value[index] = next; continue; }
      const measured = (next - this.value[index]) / elapsed;
      const previous = this.velocity[index];
      this.velocity[index] = measured * previous <= 0 || Math.abs(measured) < Math.abs(previous) ? measured : previous + (measured - previous) * 0.6;
      this.value[index] = next;
    }
    this.time = time;
  }
  sample(time: number, out: Float64Array, gains?: ArrayLike<number>): void {
    const ahead = Number.isFinite(this.time) ? Math.max(0, Math.min(time - this.time, this.interval)) : 0;
    for (let index = 0; index < this.size; index++) out[index] = this.value[index] + this.velocity[index] * ahead * (gains ? gains[index] : 1);
  }
}

/** As face-tracker.ts: emotions are held, everything else predicted; each channel kept in its range. */
const PREDICTION_GAINS = Float64Array.from(TRACKING_KEYS, (key) => key === 'happy' || key === 'angry' || key === 'surprised' ? 0 : 1);
const TRACKING_LIMITS: Record<keyof TrackingData, number> = { headYaw: 0.7, headPitch: 0.5, headRoll: 0.5, headX: 1, headY: 1, headZ: 1, mouthOpen: 1, blinkLeft: 1, blinkRight: 1, eyeLookX: 1, eyeLookY: 1, happy: 1, angry: 1, surprised: 1 };
const UNSIGNED_KEYS = new Set<keyof TrackingData>(['mouthOpen', 'blinkLeft', 'blinkRight', 'happy', 'angry', 'surprised']);

export class NativeFaceTracker {
  private settings: NativeTrackerSettings = {
    faceFps: 30, handFps: 12, handsEnabled: false, tongueEnabled: false, tongueTuning: { sensitivity: 1, inLevel: 0.08, outLevel: 0.46 }, tongueFps: 15,
    cheekTuning: { sensitivity: 1, relaxedLevel: 0, puffedLevel: 0 },
    face: { mirror: true, glassesMode: false, lowLight: false, highLight: false, stability: 0.72, eyeOpenness: 0.5, eyeCalibration: null, gazeCalibration: null },
    hands: { mirror: true, quality: 'balanced', fingerSmoothing: 0.5, lowLight: false, highLight: false },
    includeRaw: false
  };
  private provider: NativeStartOptions['provider'] = 'auto';
  private keepFrameRate = true;
  private cameraMode: NativeCameraMode | null = null;
  private latest?: NativeFrame;
  private latestTracking?: TrackingData;
  /** Face frames not yet returned by update(). */
  private pendingFace = false;
  private hands: readonly HandPose[] = [];
  private blendshapes: Readonly<Record<string, number>> = {};
  private channel?: Channel<ArrayBuffer>;
  private settingsQueued = false;
  private frames = 0;
  /** Hand quality the camera was opened with (the capture size follows it, as on the web). */
  private startedQuality: HandTrackingQuality = 'balanced';
  private readonly predictor = new MotionPredictor(TRACKING_KEYS.length);
  private readonly predictorScratch = new Float64Array(TRACKING_KEYS.length);
  private readonly predicted = {} as TrackingData;
  /** Called with every decoded frame (debug views, stats). */
  onFrame?: (frame: NativeFrame) => void;

  /** Nothing to load in the page: the models load in Rust when tracking starts. */
  async initialize(): Promise<void> {}

  /** Opens camera `cameraId` (an index, or its label from listCameras()) natively and starts tracking. */
  async start(cameraId: string | number = 0, options: NativeStartOptions = {}): Promise<NativeStatus> {
    await this.stop();
    const index = typeof cameraId === 'number' ? cameraId : await this.cameraIndex(cameraId);
    const channel = new Channel<ArrayBuffer>();
    channel.onmessage = (message) => this.receive(message);
    this.channel = channel;
    const size = CAPTURE_SIZE[this.settings.hands.quality];
    this.startedQuality = this.settings.hands.quality;
    this.predictor.reset();
    const camera = { index, width: size.width, height: size.height, fps: 60, keepFrameRate: this.keepFrameRate, ...options.camera };
    return invoke<NativeStatus>(command('start'), {
      options: { provider: this.provider, cameraMode: this.cameraMode, ...options, camera, settings: this.settings },
      onFrame: channel
    });
  }

  async stop(): Promise<void> {
    if (!this.channel) return;
    this.channel.onmessage = () => {};
    this.channel = undefined;
    await invoke(command('stop'));
    this.hands = [];
    this.blendshapes = {};
  }

  status(): Promise<NativeStatus> { return invoke<NativeStatus>(command('status')); }
  listCameras(): Promise<string[]> { return invoke<string[]>(command('list_cameras')); }
  /**
   * Every camera's exposure / gain / frame-rate controls, read without
   * streaming. Virtual cameras are not opened; physical ones are skipped while
   * another app uses a camera (Windows) unless `force`.
   */
  listCameraControls(force = false): Promise<NativeDeviceControls[]> { return invoke<NativeDeviceControls[]>(command('list_camera_controls'), { force }); }
  /**
   * The modes camera `index` offers (format, width, height, fps), read by
   * opening it without streaming; while this tracker streams that camera, the
   * modes it listed when it was opened.
   */
  listCameraFormats(index: number): Promise<NativeCameraMode[]> { return invoke<NativeCameraMode[]>(command('list_camera_formats'), { index }); }
  calibrate(): void { this.predictor.reset(); if (this.channel) void invoke(command('calibrate')); }

  /**
   * Like FaceTracker.update(): the TrackingData of a face detection that
   * arrived since the last call, or undefined. `frameSkip` is ignored (the
   * native tracker runs at its own rate, off the render loop).
   */
  update(_frameSkip = 0): TrackingData | undefined {
    if (!this.pendingFace) return undefined;
    this.pendingFace = false;
    return this.latestTracking;
  }

  /**
   * Like FaceTracker.predict(): the last face detection carried on by its
   * velocity to render time `now` (performance.now()), for at most one
   * detection interval. Returns the same object every call.
   */
  predict(now: number): TrackingData {
    const values = this.predictorScratch;
    this.predictor.sample(now, values, PREDICTION_GAINS);
    const predicted = this.predicted;
    TRACKING_KEYS.forEach((key, index) => {
      const limit = TRACKING_LIMITS[key];
      predicted[key] = Math.min(limit, Math.max(UNSIGNED_KEYS.has(key) ? 0 : -limit, values[index]));
    });
    return predicted;
  }

  getTrackingData(): TrackingData | undefined { return this.latestTracking; }
  getHands(): readonly HandPose[] { return this.hands; }
  getBlendshapes(): Readonly<Record<string, number>> { return this.blendshapes; }
  getLatestFrame(): NativeFrame | undefined { return this.latest; }
  get framesReceived(): number { return this.frames; }

  setFaceTrackingFps(fps: number): void { this.settings.faceFps = Math.max(5, Math.min(240, fps)); this.push(); }
  setHandTrackingFps(fps: number): void { this.settings.handFps = Math.max(4, Math.min(60, fps)); this.push(); }
  async setHandTracking(enabled: boolean): Promise<void> { this.settings.handsEnabled = enabled; if (!enabled) this.hands = []; this.push(); }
  setMirrorFaceTracking(enabled: boolean): void { this.settings.face.mirror = enabled; this.push(); }
  setMirrorHandTracking(enabled: boolean): void { this.settings.hands.mirror = enabled; this.push(); }
  setGlassesTrackingMode(enabled: boolean): void { this.settings.face.glassesMode = enabled; this.push(); }
  /** Settings → Eye openness: 0..1, 0.5 default (how readily a lowered lid shows). */
  setEyeOpenness(sensitivity: number): void { this.settings.face.eyeOpenness = Math.max(0, Math.min(1, sensitivity)); this.push(); }
  /** The user's eye calibration (open and closed levels per eye), or none to adapt from defaults. */
  setEyeCalibration(calibration: EyeCalibration | undefined): void { this.settings.face.eyeCalibration = calibration ?? null; this.push(); }
  /** The pupil calibration (gaze.rs); undefined goes back to the forward baseline's offset. */
  setGazeCalibration(calibration?: GazeCalibration): void { this.settings.face.gazeCalibration = calibration ?? null; this.push(); }
  /**
   * The app's low light camera fix: the face models see the frame through the
   * web tracker's adaptive brighten + blur canvas filter (reproduced natively),
   * with lower confidences, calmer smoothing and longer holds. Recalibrates on a change.
   */
  setLowLightMode(enabled: boolean): void {
    if (this.settings.face.lowLight === enabled) return;
    this.settings.face.lowLight = this.settings.hands.lowLight = enabled;
    this.push();
  }
  /** The overexposed camera fix: the darkening filter and its thresholds / smoothing. Recalibrates on a change. */
  setHighLightMode(enabled: boolean): void {
    if (this.settings.face.highLight === enabled) return;
    this.settings.face.highLight = this.settings.hands.highLight = enabled;
    this.push();
  }
  setTrackingStability(stability: number): void { this.settings.face.stability = Math.max(0, Math.min(1, stability)); this.push(); }
  setFingerSmoothing(smoothing: number): void { this.settings.hands.fingerSmoothing = Math.max(0, Math.min(1, smoothing)); this.push(); }
  setHandQuality(quality: HandTrackingQuality): void { this.settings.hands.quality = quality; this.push(); }
  setTongueTracking(enabled: boolean, tuning?: TongueTuning): void { this.settings.tongueEnabled = enabled; if (tuning) this.settings.tongueTuning = tuning; this.push(); }
  /** Cheek puff calibration; the cheek detector always runs (on the landmarks) and its puff arrives as cheekPuff. */
  setCheekTuning(tuning: CheekTuning): void { this.settings.cheekTuning = { ...tuning }; this.push(); }
  /**
   * Calibration capture: the average raw level of the next `samples` tongue
   * detections (`'tongue'`; tongue tracking must be on) or face frames
   * (`'cheeks'`), as the web detectors' captureLevel. Undefined when not
   * tracking, no face came within `timeoutMs`, or a newer capture replaced it.
   */
  async captureLevel(kind: CaptureKind, samples = 30, timeoutMs = 6000): Promise<number | undefined> {
    if (!this.channel) return undefined;
    const level = await invoke<number | null>(command('capture_level'), { kind, samples, timeoutMs });
    return level ?? undefined;
  }
  /**
   * "Keep frame rate" (default on): stops the webcam from lowering its frame
   * rate in low light (auto-exposure priority off, else an exposure cap with
   * more gain). Applies live to a running camera and to the next start();
   * the camera's own settings are restored when tracking stops or this is
   * turned off. Virtual cameras are left alone. See status().cameraControls.
   */
  setKeepFrameRate(enabled: boolean): void {
    if (this.keepFrameRate === enabled) return;
    this.keepFrameRate = enabled;
    if (this.channel) void invoke(command('set_keep_frame_rate'), { enabled });
  }
  getKeepFrameRate(): boolean { return this.keepFrameRate; }
  /**
   * The camera mode of the next start() (one of listCameraFormats()), or null
   * for Auto: the best format for the hand quality's size (MJPEG first, never
   * below 15 fps). Changing it needs a restart (the camera reopens).
   */
  setCameraMode(mode: NativeCameraMode | null): void { this.cameraMode = mode ? { ...mode } : null; }
  getCameraMode(): NativeCameraMode | null { return this.cameraMode; }
  /** 'GPU' = Auto (GPU while it is faster, CPU under contention), 'CPU' = CPU only. Applies on the next start(). */
  setDelegate(delegate: 'GPU' | 'CPU'): void { this.provider = delegate === 'CPU' ? 'cpu' : 'auto'; }
  /** The web version may run hands in a worker; natively they always run on their own thread. */
  setHandWorker(_allowed: boolean): void {}
  /** The web version picks where detection runs; natively it always has its own threads. */
  setDetectionThread(_thread: string): void {}
  setWorkerFallback(_enabled: boolean): void {}
  /** Whether the open camera was asked for another size than `quality` wants (the caller may restart). */
  captureSizeDiffers(quality: HandTrackingQuality): boolean {
    if (!this.channel) return false;
    const opened = CAPTURE_SIZE[this.startedQuality], wanted = CAPTURE_SIZE[quality];
    return opened.width !== wanted.width || opened.height !== wanted.height;
  }

  private receive(message: ArrayBuffer): void {
    const frame = decodeFrame(message);
    this.frames++;
    this.latest = frame;
    if (frame.faceUpdated && frame.tracking) {
      this.latestTracking = frame.tracking;
      this.pendingFace = true;
      // The detection belongs to the camera frame taken latencyMs ago.
      const values = this.predictorScratch;
      const tracking = frame.tracking;
      TRACKING_KEYS.forEach((key, index) => { values[index] = tracking[key]; });
      this.predictor.push(values, performance.now() - frame.latencyMs);
    }
    this.blendshapes = frame.blendshapes;
    if (frame.handsUpdated || frame.hands.length !== this.hands.length) this.hands = frame.hands;
    this.onFrame?.(frame);
  }

  /** Several set* calls in a row reach Rust as one update. */
  private push(): void {
    if (this.settingsQueued) return;
    this.settingsQueued = true;
    queueMicrotask(() => {
      this.settingsQueued = false;
      if (this.channel) void invoke(command('update_settings'), { settings: this.settings });
    });
  }

  private async cameraIndex(label: string): Promise<number> {
    if (!label) return 0;
    const cameras = await this.listCameras();
    const found = cameras.findIndex((camera) => camera.includes(label));
    return found >= 0 ? Number.parseInt(cameras[found], 10) || found : 0;
  }
}
