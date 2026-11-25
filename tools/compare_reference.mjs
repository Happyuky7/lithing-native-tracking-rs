// Compares `lithing-track dump` output with the web MediaPipe reference
// (tools/webref.mjs) frame by frame: raw face / hand landmarks and blendshapes,
// and, when both sides carry them, the post-processed TrackingData (the app's
// FaceTracker vs the Rust port, same mode) and the light filter used per frame.
//   node tools/compare_reference.mjs <native.jsonl> <reference.jsonl> <width> <height> [--json out.json]
import fs from 'node:fs';

const [nativeFile, referenceFile, width, height] = process.argv.slice(2);
const jsonOut = process.argv.includes('--json') ? process.argv[process.argv.indexOf('--json') + 1] : null;
const W = Number(width), H = Number(height);
const read = (file) => fs.readFileSync(file, 'utf8').trim().split('\n').map((line) => JSON.parse(line));
const native = new Map(read(nativeFile).map((f) => [f.frame, f]));
const reference = read(referenceFile);
const NAMES = ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight'];

const px = (p) => [p[0] * W, p[1] * H];
const dist2 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN);
const pct = (a, q) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.min(s.length - 1, Math.floor(s.length * q))] : NaN; };

let faceBoth = 0, faceNativeOnly = 0, faceRefOnly = 0;
const faceErrPx = [], faceErrIod = [], irisErrPx = [];
const blendDiff = NAMES.map(() => []);
let handPairs = 0, handNativeCount = 0, handRefCount = 0, labelAgree = 0;
const handErrPx = [], handErrPalm = [], worldErrMm = [];
const TRACKING_KEYS = ['headYaw', 'headPitch', 'headRoll', 'headX', 'headY', 'headZ', 'mouthOpen', 'blinkLeft', 'blinkRight', 'eyeLookX', 'eyeLookY', 'happy', 'angry', 'surprised'];
const trackingDiff = Object.fromEntries(TRACKING_KEYS.map((k) => [k, []]));
let trackingPairs = 0, filterSame = 0, filterFrames = 0;

for (const ref of reference) {
  const nat = native.get(ref.frame);
  if (!nat) continue;
  if (ref.face && nat.face) {
    faceBoth++;
    const a = ref.face.landmarks, b = nat.face.landmarks;
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += dist2(px(a[i]), px(b[i]));
    const err = sum / a.length;
    const iod = dist2(px(a[33]), px(a[263]));
    faceErrPx.push(err);
    faceErrIod.push(err / iod * 100);
    irisErrPx.push((dist2(px(a[468]), px(b[468])) + dist2(px(a[473]), px(b[473]))) / 2);
    for (let k = 0; k < 52; k++) blendDiff[k].push(Math.abs(ref.face.blendshapes[k] - nat.face.blendshapes[k]));
  } else if (ref.face) faceRefOnly++;
  else if (nat.face) faceNativeOnly++;
  if (ref.tracking && nat.tracking) {
    trackingPairs++;
    for (const key of TRACKING_KEYS) trackingDiff[key].push(Math.abs(ref.tracking[key] - nat.tracking[key]));
  }
  if (ref.filter !== undefined && ref.filter !== null) {
    filterFrames++;
    if (nat.light && nat.light.filter === ref.filter) filterSame++;
  }
  const rh = ref.hands ?? [], nh = nat.hands ?? [];
  handRefCount += rh.length;
  handNativeCount += nh.length;
  // Pair hands by wrist distance (at most two each).
  const used = new Set();
  for (const r of rh) {
    let best = -1, bestDistance = Infinity;
    nh.forEach((n, j) => { if (used.has(j)) return; const d = dist2(px(r.landmarks[0]), px(n.landmarks[0])); if (d < bestDistance) { bestDistance = d; best = j; } });
    const palm = dist2(px(r.landmarks[0]), px(r.landmarks[9]));
    if (best < 0 || bestDistance > palm * 1.5) continue;
    used.add(best);
    const n = nh[best];
    handPairs++;
    let sum = 0, world = 0;
    for (let i = 0; i < 21; i++) {
      sum += dist2(px(r.landmarks[i]), px(n.landmarks[i]));
      world += Math.hypot(r.world[i][0] - n.world[i][0], r.world[i][1] - n.world[i][1], r.world[i][2] - n.world[i][2]);
    }
    handErrPx.push(sum / 21);
    handErrPalm.push(sum / 21 / palm * 100);
    worldErrMm.push(world / 21 * 1000);
    if (r.label === n.label) labelAgree++;
  }
}

const blendRows = NAMES.map((name, k) => ({ name, mad: mean(blendDiff[k]), p95: pct(blendDiff[k], 0.95), max: Math.max(...blendDiff[k]) }));
const summary = {
  frames: reference.length,
  face: { both: faceBoth, nativeOnly: faceNativeOnly, referenceOnly: faceRefOnly, landmarkErrorPx: { mean: mean(faceErrPx), p95: pct(faceErrPx, 0.95) }, landmarkErrorPercentIOD: { mean: mean(faceErrIod), p95: pct(faceErrIod, 0.95) }, irisErrorPx: mean(irisErrPx), blendshapeMAD: mean(blendRows.map((r) => r.mad)), blendshapeMaxMAD: blendRows.reduce((a, b) => (b.mad > a.mad ? b : a)) },
  hands: { reference: handRefCount, native: handNativeCount, matched: handPairs, landmarkErrorPx: { mean: mean(handErrPx), p95: pct(handErrPx, 0.95) }, landmarkErrorPercentPalm: mean(handErrPalm), worldErrorMm: mean(worldErrMm), labelAgreement: handPairs ? labelAgree / handPairs : NaN },
  blendshapes: blendRows,
  detectionRate: { native: (faceBoth + faceNativeOnly) / reference.length, reference: (faceBoth + faceRefOnly) / reference.length },
  tracking: trackingPairs ? { frames: trackingPairs, meanAbs: Object.fromEntries(TRACKING_KEYS.map((k) => [k, +mean(trackingDiff[k]).toFixed(5)])), p95: Object.fromEntries(TRACKING_KEYS.map((k) => [k, +pct(trackingDiff[k], 0.95).toFixed(5)])), max: Object.fromEntries(TRACKING_KEYS.map((k) => [k, +Math.max(...trackingDiff[k]).toFixed(5)])) } : undefined,
  lightFilter: filterFrames ? { framesWithSameFilter: filterSame, frames: filterFrames } : undefined,
};
console.log(JSON.stringify({ ...summary, blendshapes: undefined }, null, 1));
console.log('worst blendshapes (MAD):', blendRows.slice().sort((a, b) => b.mad - a.mad).slice(0, 8).map((r) => `${r.name} ${r.mad.toFixed(4)}`).join(', '));
if (jsonOut) fs.writeFileSync(jsonOut, JSON.stringify(summary, null, 1));
