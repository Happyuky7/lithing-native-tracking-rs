/**
 * Decoder of the native tracker's per-frame payload (see src/wire.rs for the
 * layout). Pure TypeScript with no imports, so it runs in the web view, in a
 * worker and in Node (the round-trip test runs it with type stripping).
 */

export type Vec3 = [number, number, number];
export type HandGesture = 'none' | 'open' | 'fist' | 'victory' | 'thumbs-up' | 'thumbs-down' | 'pointing' | 'ok' | 'rock' | 'love-you' | 'call-me' | 'finger-heart';

/** Same fields and meaning as TrackingData in src/app/face-tracker.ts. */
export interface TrackingData {
  headYaw: number; headPitch: number; headRoll: number;
  headX: number; headY: number; headZ: number;
  mouthOpen: number; blinkLeft: number; blinkRight: number;
  eyeLookX: number; eyeLookY: number;
  happy: number; angry: number; surprised: number;
}

export interface Landmark { x: number; y: number; z: number }

/** Same fields and meaning as HandPose in src/app/face-tracker.ts. */
export interface HandPose {
  handedness: 'left' | 'right';
  userHand: 'left' | 'right';
  wristX: number; wristY: number; wristDepth: number;
  faceX: number; faceY: number;
  direction: Vec3; across: Vec3;
  fingers: Float32Array;
  gesture: HandGesture;
  gestureConfidence: number;
  landmarks?: readonly Landmark[];
}

/** One eye's openness cues and levels (see eye-openness.ts in the app, eyes.rs in lithing-tracking). */
export interface NativeEye {
  /** Raw eyeBlink blendshape, lid gap over eye width, lid gap over iris diameter. */
  blink: number; gap: number; iris: number;
  /** Geometric openness 0 closed .. 1 open, fused closure, the closure sent on. */
  geometry: number; fused: number; output: number;
  /** The lid gap range in use and whether it starts from the eye calibration. */
  gapOpen: number; gapClosed: number; calibrated: boolean;
}

export interface NativeFrame {
  frameId: number;
  /** Capture time on the tracker's clock (ms). */
  timestamp: number;
  latencyMs: number;
  faceMs: number;
  handsMs: number;
  /** A face detection ran on this frame (the web update() returns TrackingData only then). */
  faceUpdated: boolean;
  handsUpdated: boolean;
  tracking?: TrackingData;
  /** Avatar-side names, like FaceTracker.getBlendshapes(); empty without a face. */
  blendshapes: Record<string, number>;
  hands: HandPose[];
  faceLandmarks?: Landmark[];
  /** The user's right and left eye, while a face is tracked (newer plugins only). */
  eyes?: { right: NativeEye; left: NativeEye };
  /** Raw gaze before baseline and mirroring (gaze.ts in the app, gaze.rs in lithing-tracking), while a face is tracked (newer plugins only). */
  gaze?: { x: number; y: number };
  providers: { face: 'cpu' | 'directml' | 'cuda' | 'nnapi'; hands: 'cpu' | 'directml' | 'cuda' | 'nnapi' };
}

export const TRACKING_KEYS: readonly (keyof TrackingData)[] = ['headYaw', 'headPitch', 'headRoll', 'headX', 'headY', 'headZ', 'mouthOpen', 'blinkLeft', 'blinkRight', 'eyeLookX', 'eyeLookY', 'happy', 'angry', 'surprised'];
export const BLENDSHAPE_CHANNELS: readonly string[] = [
  '_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight',
  'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft',
  'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft',
  'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
  'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight',
  'tongueOut', 'tongueUp', 'tongueDown', 'tongueLeft', 'tongueRight'
];
export const GESTURES: readonly HandGesture[] = ['none', 'open', 'fist', 'victory', 'thumbs-up', 'thumbs-down', 'pointing', 'ok', 'rock', 'love-you', 'call-me', 'finger-heart'];

const MAGIC = 0x3146544c; // "LTF1" little-endian
const FLAG_TRACKING = 1, FLAG_FACE_UPDATED = 2, FLAG_HANDS_UPDATED = 4, FLAG_BLENDSHAPES = 8, FLAG_TONGUE = 16, FLAG_HAND_LANDMARKS = 32, FLAG_FACE_LANDMARKS = 64, FLAG_EYES = 128, FLAG_GAZE = 256;
const TRACKING_OFFSET = 40, BLENDSHAPES_OFFSET = 96, HANDS_OFFSET = 324, HAND_SIZE = 132;
const PROVIDERS = ['cpu', 'directml', 'cuda', 'nnapi'] as const;

/** Decodes one payload; throws on anything that is not a version-1 frame. */
export function decodeFrame(buffer: ArrayBuffer | ArrayBufferView): NativeFrame {
  const view = buffer instanceof ArrayBuffer ? new DataView(buffer) : new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  if (view.byteLength < HANDS_OFFSET || view.getUint32(0, true) !== MAGIC) throw new Error('not a native tracking frame');
  if (view.getUint16(4, true) !== 1) throw new Error(`unsupported frame version ${view.getUint16(4, true)}`);
  const flags = view.getUint16(6, true);
  const f = (offset: number) => view.getFloat32(offset, true);
  const count = view.getUint8(36);
  let tracking: TrackingData | undefined;
  if (flags & FLAG_TRACKING) {
    const values = {} as TrackingData;
    TRACKING_KEYS.forEach((key, index) => { values[key] = f(TRACKING_OFFSET + index * 4); });
    tracking = values;
  }
  const blendshapes: Record<string, number> = {};
  if (flags & FLAG_BLENDSHAPES) {
    // The tongue channels exist only while the tongue detector reports (like the web record).
    const channels = flags & FLAG_TONGUE ? BLENDSHAPE_CHANNELS.length : 52;
    for (let index = 0; index < channels; index++) blendshapes[BLENDSHAPE_CHANNELS[index]] = f(BLENDSHAPES_OFFSET + index * 4);
  }
  const hands: HandPose[] = [];
  for (let h = 0; h < count; h++) {
    const o = HANDS_OFFSET + h * HAND_SIZE;
    const v = (i: number) => f(o + 4 + i * 4);
    const fingers = new Float32Array(20);
    for (let i = 0; i < 20; i++) fingers[i] = v(11 + i);
    hands.push({
      handedness: view.getUint8(o) === 1 ? 'right' : 'left',
      userHand: view.getUint8(o + 1) === 1 ? 'right' : 'left',
      gesture: GESTURES[view.getUint8(o + 2)] ?? 'none',
      wristX: v(0), wristY: v(1), wristDepth: v(2), faceX: v(3), faceY: v(4),
      direction: [v(5), v(6), v(7)], across: [v(8), v(9), v(10)],
      fingers, gestureConfidence: v(31)
    });
  }
  let offset = HANDS_OFFSET + count * HAND_SIZE;
  const points = (n: number): Landmark[] => {
    const out: Landmark[] = [];
    for (let i = 0; i < n; i++, offset += 12) out.push({ x: f(offset), y: f(offset + 4), z: f(offset + 8) });
    return out;
  };
  if (flags & FLAG_HAND_LANDMARKS) for (const hand of hands) hand.landmarks = points(21);
  const faceLandmarks = flags & FLAG_FACE_LANDMARKS ? points(478) : undefined;
  const eye = (): NativeEye => {
    const v = (i: number) => f(offset + i * 4);
    const value: NativeEye = { blink: v(0), gap: v(1), iris: v(2), geometry: v(3), fused: v(4), output: v(5), gapOpen: v(6), gapClosed: v(7), calibrated: v(8) !== 0 };
    offset += 36;
    return value;
  };
  const eyes = flags & FLAG_EYES && view.byteLength >= offset + 72 ? { right: eye(), left: eye() } : undefined;
  // The gaze follows the eyes; it is only read when the eyes were (a frame never has it without them).
  const gaze = flags & FLAG_GAZE && (eyes || !(flags & FLAG_EYES)) && view.byteLength >= offset + 8 ? { x: f(offset), y: f(offset + 4) } : undefined;
  return {
    frameId: view.getFloat64(8, true),
    timestamp: view.getFloat64(16, true),
    latencyMs: f(24), faceMs: f(28), handsMs: f(32),
    faceUpdated: Boolean(flags & FLAG_FACE_UPDATED),
    handsUpdated: Boolean(flags & FLAG_HANDS_UPDATED),
    tracking, blendshapes, hands, faceLandmarks, eyes, gaze,
    providers: { face: PROVIDERS[view.getUint8(37)] ?? 'cpu', hands: PROVIDERS[view.getUint8(38)] ?? 'cpu' }
  };
}
