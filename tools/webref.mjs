// Usage: node tools/webref.mjs <clip> <out.jsonl> [CPU|GPU] [options]
//   options: --raw <file.rgb> --size WxH   raw RGB24 frames instead of $LITHING_FRAMES/<clip>/<n>.png
//            --fps N                       frame rate for the timestamps (default 30)
//            --frames N                    stop after N frames
//            --mode none|low|high          the app's low / high light mode (setLowLightMode / setHighLightMode)
//            --glasses                     setGlassesTrackingMode(true)
// PNG frames come from ffmpeg -i clip.y4m -start_number 0 <dir>/%d.png; raw frames from
// ffmpeg -i clip -f rawvideo -pix_fmt rgb24 clip.rgb (the same pixels `lithing-track dump --raw-size` reads).
// The MediaPipe package, .task files and src/app/*.ts come from the main app (LITHING_WEB_REPO, default ../..).
// Needs puppeteer-core (npm i puppeteer-core) and Chrome; runs headless.
//
// Runs the main repo's MediaPipe (same package version, same .task models, same options as
// src/app/face-tracker.ts) headlessly in Chrome, VIDEO mode, and writes raw outputs. With a mode,
// the face sees the app's light-filtered canvas (the lines of FaceTracker.update(), on the
// tracker's own canvases and brightness state) and the app's FaceTracker turns every face result
// into TrackingData (processFace: calibration, glasses weights, smoothing, step limits, holds),
// which is written as `tracking` next to the raw landmarks. src/app/*.ts is served with Node's
// type stripping; nothing in the app is modified.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { stripTypeScriptTypes } from 'node:module';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
const argv = process.argv.slice(2);
const flag = (name, fallback) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : fallback; };
const VALUE_FLAGS = new Set(['--raw', '--size', '--fps', '--frames', '--mode']);
const [clip, outFile, delegate = 'CPU'] = argv.filter((a, i) => !a.startsWith('--') && !VALUE_FLAGS.has(argv[i - 1]));
const raw = flag('--raw');
const [rawW, rawH] = (flag('--size', '640x480')).split('x').map(Number);
const fps = Number(flag('--fps', '30'));
const mode = flag('--mode', 'none');
const glasses = argv.includes('--glasses');
const here = path.dirname(fileURLToPath(import.meta.url));
const REPO = process.env.LITHING_WEB_REPO ?? path.resolve(here, '../..');
const FRAMES = process.env.LITHING_FRAMES ?? `${here}/frames`;
const routes = { '/vision/': `${REPO}/node_modules/@mediapipe/tasks-vision/`, '/wasm/': `${REPO}/public/mediapipe/wasm/`, '/models/': `${REPO}/public/mediapipe/`, '/frames/': FRAMES + '/' };
const types = { '.mjs': 'text/javascript', '.js': 'text/javascript', '.wasm': 'application/wasm', '.png': 'image/png', '.html': 'text/html' };
const frameBytes = rawW * rawH * 3;
const rawFd = raw ? fs.openSync(raw, 'r') : null;
const server = http.createServer((req, res) => {
  const url = decodeURIComponent(req.url.split('?')[0]);
  const headers = (type) => ({ 'content-type': type, 'cross-origin-opener-policy': 'same-origin', 'cross-origin-embedder-policy': 'require-corp' });
  if (url === '/') { res.writeHead(200, { 'content-type': 'text/html' }); res.end('<!doctype html><title>ref</title>'); return; }
  if (url.startsWith('/raw/')) {
    const buffer = Buffer.alloc(frameBytes);
    fs.readSync(rawFd, buffer, 0, frameBytes, Number(url.slice(5)) * frameBytes);
    res.writeHead(200, headers('application/octet-stream')); res.end(buffer); return;
  }
  if (url.startsWith('/app/')) {
    // The app's TypeScript modules, types stripped (imports have no extension).
    const file = `${REPO}/src/app/${url.slice(5).replace(/\.ts$/, '')}.ts`;
    if (!fs.existsSync(file)) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, headers('text/javascript'));
    res.end(stripTypeScriptTypes(fs.readFileSync(file, 'utf8'), { mode: 'strip' }));
    return;
  }
  let file = null;
  for (const [prefix, dir] of Object.entries(routes)) if (url.startsWith(prefix)) file = dir + url.slice(prefix.length);
  if (!file || !fs.existsSync(file)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, headers(types[path.extname(file)] ?? 'application/octet-stream'));
  fs.createReadStream(file).pipe(res);
}).listen(0, '127.0.0.1');
await new Promise(r => server.once('listening', r));
const port = server.address().port;
let count = raw ? Math.floor(fs.statSync(raw).size / frameBytes) : fs.readdirSync(`${FRAMES}/${clip}`).length;
if (flag('--frames')) count = Math.min(count, Number(flag('--frames')));
const browser = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, args: ['--enable-unsafe-swiftshader', '--use-angle=d3d11'] });
const page = await browser.newPage();
page.on('console', m => console.error('page:', m.text()));
page.on('pageerror', e => console.error('page error:', e.message));
await page.goto(`http://127.0.0.1:${port}/`);
const out = fs.createWriteStream(outFile);
await page.exposeFunction('emit', line => out.write(line + '\n'));
const result = await page.evaluate(async (clip, count, delegate, raw, W, H, fps, mode, glasses) => {
  const vision = await import('/vision/vision_bundle.mjs');
  const { FaceTracker } = await import('/app/face-tracker');
  // The app's tracker, used for its settings, light state and post-processing (no camera, no landmarker of its own).
  const tracker = new FaceTracker();
  tracker.setLowLightMode(mode === 'low');
  tracker.setHighLightMode(mode === 'high');
  tracker.setGlassesTrackingMode(glasses);
  tracker.calibrate();
  const files = await vision.FilesetResolver.forVisionTasks('/wasm');
  const face = await vision.FaceLandmarker.createFromOptions(files, { baseOptions: { modelAssetPath: '/models/face_landmarker.task', delegate }, runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true, ...tracker.faceConfidenceOptions() });
  const hands = await vision.HandLandmarker.createFromOptions(files, { baseOptions: { modelAssetPath: '/models/hand_landmarker.task', delegate }, runningMode: 'VIDEO', numHands: 2, minHandDetectionConfidence: 0.4, minHandPresenceConfidence: 0.4, minTrackingConfidence: 0.4 });
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  const faceMs = [], handMs = [];
  const r6 = v => Math.round(v * 1e6) / 1e6;
  const rgba = raw ? new Uint8ClampedArray(W * H * 4) : null;
  for (let i = 0; i < count; i++) {
    let image;
    if (raw) {
      const rgb = new Uint8Array(await (await fetch(`/raw/${i}`)).arrayBuffer());
      for (let p = 0, q = 0; p < rgb.length; p += 3, q += 4) { rgba[q] = rgb[p]; rgba[q + 1] = rgb[p + 1]; rgba[q + 2] = rgb[p + 2]; rgba[q + 3] = 255; }
      image = await createImageBitmap(new ImageData(rgba, W, H));
    } else {
      const blob = await (await fetch(`/frames/${clip}/${i}.png`)).blob();
      image = await createImageBitmap(blob);
    }
    const ts = i * 1000 / fps;
    // ---- FaceTracker.update(), the light processing (face-tracker.ts lines 908-925), on the tracker's own state ----
    let input = image;
    let filter = null;
    tracker.video = image;
    if ((tracker.lowLightMode || tracker.highLightMode) && tracker.processingContext) {
      tracker.processingCanvas.width = 640;
      tracker.processingCanvas.height = 480;
      if (tracker.lowLightMode) {
        const luminance = tracker.measureCenterLuminance();
        const targetBrightness = clamp(0.42 / Math.max(0.08, luminance), 1.12, 2.45);
        tracker.lowLightBrightness += (targetBrightness - tracker.lowLightBrightness) * 0.16;
        const contrast = clamp(1.04 + Math.max(0, 0.34 - luminance) * 0.48, 1.04, 1.2);
        const blur = tracker.lowLightBrightness > 1.8 ? 0.65 : 0.42;
        tracker.processingContext.filter = `blur(${blur.toFixed(2)}px) brightness(${tracker.lowLightBrightness.toFixed(2)}) contrast(${contrast.toFixed(2)}) saturate(0.78)`;
      } else {
        tracker.processingContext.filter = 'brightness(0.72) contrast(1.1) saturate(0.9)';
      }
      filter = tracker.processingContext.filter;
      tracker.processingContext.drawImage(image, 0, 0, 640, 480);
      tracker.processingContext.filter = 'none';
      input = tracker.processingCanvas;
    }
    // ----
    let t = performance.now(); const f = face.detectForVideo(input, ts); faceMs.push(performance.now() - t);
    // Hands see the camera frame as it is (detectHandsHere).
    t = performance.now(); const h = hands.detectForVideo(image, ts); handMs.push(performance.now() - t);
    const tracking = tracker.processFace(f, ts, ts);
    image.close();
    const line = { frame: i, filter,
      face: f.faceLandmarks[0] ? { landmarks: f.faceLandmarks[0].map(p => [r6(p.x), r6(p.y), r6(p.z)]), blendshapes: f.faceBlendshapes[0].categories.map(c => r6(c.score)), names: i === 0 ? f.faceBlendshapes[0].categories.map(c => c.categoryName) : undefined } : null,
      hands: h.landmarks.map((l, k) => ({ landmarks: l.map(p => [r6(p.x), r6(p.y), r6(p.z)]), world: h.worldLandmarks[k].map(p => [r6(p.x), r6(p.y), r6(p.z)]), label: h.handedness[k][0].categoryName.toLowerCase(), labelScore: h.handedness[k][0].score })),
      tracking, blendshapesAvatar: tracker.getBlendshapes() };
    await window.emit(JSON.stringify(line));
  }
  const stat = a => { const s = [...a].slice(5).sort((x, y) => x - y); return { mean: s.reduce((x, y) => x + y, 0) / s.length, p50: s[s.length >> 1], p95: s[Math.floor(s.length * 0.95)] }; };
  return { faceMs: stat(faceMs), handMs: stat(handMs) };
}, clip, count, delegate, Boolean(raw), rawW, rawH, fps, mode, glasses);
console.log(JSON.stringify({ clip, delegate, mode, glasses, ...result }));
out.end();
await browser.close();
server.close();
