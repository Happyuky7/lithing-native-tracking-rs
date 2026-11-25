// Quick test bridge: native Vtuber-Lithing tracking -> VRChat over OSC (no VRCFaceTracking).
// Sends VRChat's native eye tracking plus Unified Expressions v2 parameters in whatever form the
// avatar declares them: it reads VRChat's OSC config of the avatar being worn
// (LocalLow/VRChat/VRChat/OSC/<user>/Avatars/<avatar>.json), finds the face tracking prefix
// (FT/v2/, v2/, …) and sends each expression as a float or as binary bits (1/2/4/8 + Negative),
// exactly as that avatar expects.
//
// Usage (close Vtuber-Lithing first, the camera can only have one owner):
//   node tools/vrchat-osc-bridge.mjs [cameraIndex] [--no-tongue] [--no-low-light] [--port 9000] [--avatar <avtr_id>]
// Enable OSC in VRChat (Action menu > Options > OSC > Enabled). The avatar is picked from VRChat's
// /avatar/change message (listened to on 9001 when free), else the most recently used avatar config.
import { spawn } from 'node:child_process';
import { createSocket } from 'node:dgram';
import { closeSync, existsSync, openSync, readdirSync, readFileSync, readSync, rmSync, statSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const args = process.argv.slice(2);
const option = (name) => (args.includes(name) ? args[args.indexOf(name) + 1] : undefined);
const camera = Number(args.find((arg, index) => /^\d+$/.test(arg) && !['--port'].includes(args[index - 1])) ?? 0);
// A .y4m path instead of a camera index replays a file (for testing without the webcam).
const file = args.find((arg) => arg.endsWith('.y4m'));
const tongue = !args.includes('--no-tongue');
// Low light is on unless asked otherwise: a dark room is the usual case and costs ~1 ms.
const lowLight = !args.includes('--no-low-light');
const port = Number(option('--port')) || 9000;
const root = resolve(import.meta.dirname, '..');
const exe = join(root, 'target', 'release', process.platform === 'win32' ? 'lithing-track.exe' : 'lithing-track');
const out = join(tmpdir(), `lithing-vrchat-${process.pid}.jsonl`);

// ---- The avatar's declared OSC inputs.
const oscRoot = join(homedir(), 'AppData', 'LocalLow', 'VRChat', 'VRChat', 'OSC');
/** name -> 'Float' | 'Bool' | 'Int' for every parameter the avatar accepts as input. */
let inputs = new Map();
let prefix = 'FT/v2/';
let avatarName = '(none)';
/** VRCFury-prefixed copies of a parameter, by original name. */
let aliases = new Map();
function avatarConfigs() {
  if (!existsSync(oscRoot)) return [];
  const files = [];
  for (const user of readdirSync(oscRoot)) {
    const folder = join(oscRoot, user, 'Avatars');
    if (!existsSync(folder)) continue;
    for (const name of readdirSync(folder)) if (name.endsWith('.json')) files.push(join(folder, name));
  }
  return files;
}
function loadAvatar(id) {
  const files = avatarConfigs();
  const path = id ? files.find((f) => f.endsWith(`${id}.json`)) : files.sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs)[0];
  if (!path) { console.log(`\nno VRChat OSC config found${id ? ` for ${id}` : ''}; sending ${prefix}… floats`); return; }
  const config = JSON.parse(readFileSync(path, 'utf8').replace(/^﻿/, ''));
  inputs = new Map(config.parameters.filter((p) => p.input).map((p) => [p.name, p.input.type]));
  // The face tracking prefix: whatever precedes a known Unified Expressions name.
  const sample = [...inputs.keys()].find((name) => /(^|\/)v2\/(JawOpen|EyeLidLeft|EyeLeftX|MouthUpperUp)/.test(name) && !name.startsWith('VF'));
  prefix = sample ? sample.slice(0, sample.indexOf('v2/') + 3) : 'v2/';
  avatarName = config.name ?? config.id;
  aliases = new Map();
  for (const name of inputs.keys()) {
    const match = /^VF\d+_(.+)$/.exec(name);
    if (match) aliases.set(match[1], [...(aliases.get(match[1]) ?? []), name]);
  }
  const face = [...inputs.keys()].filter((name) => name.startsWith(prefix)).length;
  console.log(`\navatar: ${avatarName} · prefix ${prefix} · ${face} face tracking inputs`);
}
loadAvatar(option('--avatar'));
// VRChat announces avatar changes on its output port; follow them when the port is free.
const listener = createSocket('udp4');
listener.on('message', (message) => {
  const address = message.toString('latin1', 0, message.indexOf(0));
  if (address !== '/avatar/change') return;
  const start = message.indexOf(',s') + 4;
  const id = message.toString('latin1', start, message.indexOf(0, start));
  if (id) setTimeout(() => loadAvatar(id), 500);
});
listener.on('error', () => listener.close());
listener.bind(9001, '127.0.0.1');

// ---- OSC encoding (UDP, one message per datagram).
const socket = createSocket('udp4');
const pad = (buffer) => Buffer.concat([buffer, Buffer.alloc(4 - (buffer.length % 4))]);
function send(address, ...values) {
  const tags = ',' + values.map((value) => (typeof value === 'boolean' ? (value ? 'T' : 'F') : 'f')).join('');
  const data = values.filter((value) => typeof value !== 'boolean').map((value) => { const b = Buffer.alloc(4); b.writeFloatBE(value); return b; });
  socket.send(Buffer.concat([pad(Buffer.from(address)), pad(Buffer.from(tags)), ...data]), port, '127.0.0.1');
  sent++;
}
const clamp = (value, low = 0, high = 1) => Math.min(high, Math.max(low, value));
/** Last binary step sent per parameter (see expression). */
const levels = new Map();
/**
 * One Unified Expressions value (0..1, or -1..1 for combined ones) to the avatar in its declared
 * form: a float parameter, or binary bits name1/2/4/8 (+ nameNegative). Without a config, a float.
 */
function expression(name, value) {
  const full = prefix + name;
  if (inputs.size === 0 || inputs.get(full) === 'Float') {
    send(`/avatar/parameters/${full}`, value);
    // VRCFury renames copies of some parameters (VF57_FT/v2/JawOpen…); feed those too.
    for (const alias of aliases.get(full) ?? []) send(`/avatar/parameters/${alias}`, value);
    return;
  }
  const bits = [1, 2, 4, 8, 16].filter((bit) => inputs.has(`${full}${bit}`));
  if (!bits.length) return;
  const max = bits.reduce((a, b) => a + b, 0);
  // Hysteresis per parameter: a step only changes once the value is well past it, so noise
  // around a boundary doesn't make the avatar flicker between two shapes.
  const scaled = Math.min(max, Math.abs(value) * max) * (value < 0 ? -1 : 1);
  const previous = levels.get(full) ?? 0;
  const held = Math.abs(scaled - previous) < 0.7 ? previous : Math.round(scaled);
  levels.set(full, held);
  const level = Math.abs(held);
  for (const bit of bits) send(`/avatar/parameters/${full}${bit}`, (level & bit) !== 0);
  if (inputs.has(`${full}Negative`)) send(`/avatar/parameters/${full}Negative`, held < 0);
}
function flag(name, value) {
  const type = inputs.get(name);
  if (inputs.size && !type) return;
  send(`/avatar/parameters/${name}`, type === 'Float' || (!type && typeof value === 'number') ? Number(value) : Boolean(value));
}

// ---- Native tracker writing one JSON line per frame.
const childArgs = ['run', file ?? `camera:${camera}`, '--ep', 'auto', '--out', out, '--quiet', ...(tongue ? ['--tongue'] : []), ...(lowLight ? ['--low-light'] : [])];
if (!existsSync(exe)) { console.error(`Not built: ${exe}\nRun: cargo build --release`); process.exit(1); }
const child = spawn(exe, childArgs, { stdio: ['ignore', 'inherit', 'inherit'] });
child.on('exit', (code) => { console.log(`\ntracker exited (${code})`); stop(); });
console.log(`${file ?? `camera:${camera}`} -> VRChat OSC 127.0.0.1:${port}${tongue ? ' · tongue on' : ''}${lowLight ? ' · low light' : ''}. Ctrl+C to stop.`);

let offset = 0;
let pending = '';
let frames = 0;
let faces = 0;
let sent = 0;
const timer = setInterval(() => {
  if (!existsSync(out)) return;
  const size = statSync(out).size;
  if (size <= offset) return;
  const fd = openSync(out, 'r');
  const chunk = Buffer.alloc(size - offset);
  readSync(fd, chunk, 0, chunk.length, offset);
  closeSync(fd);
  offset = size;
  pending += chunk.toString('utf8');
  const lines = pending.split('\n');
  pending = lines.pop() ?? '';
  // Only the newest frame matters for the avatar; older ones still count for the stats.
  const usable = lines.filter(Boolean);
  frames += usable.length;
  const last = usable.pop();
  if (last) { try { if (apply(JSON.parse(last))) faces += usable.length + 1; } catch { /* partial line */ } }
}, 8);
const status = setInterval(() => {
  process.stdout.write(`\rframes ${frames}/s · face ${faces}/s · OSC ${sent}/s · ${avatarName}   `);
  frames = 0;
  faces = 0;
  sent = 0;
}, 1000);

const tongueState = { level: 0, out: false, x: 0, y: 0 };
const neutral = { samples: 0, sum: {}, rest: {} };
const smoothed = {};
const eyes = { x: 0, y: 0, lidLeft: 0.75, lidRight: 0.75 };

function apply(frame) {
  const raw = frame.blendshapes ?? frame.face?.blendshapes;
  if (!raw || Object.keys(raw).length === 0) return false;
  const tracking = frame.tracking ?? {};
  // Neutral face: the first ~1.5 s with a face are averaged and subtracted (capped), like the
  // app's neutral calibration, so a resting face (camera below it, glasses) sends a rest pose.
  if (neutral.samples < 45) {
    for (const [name, value] of Object.entries(raw)) neutral.sum[name] = (neutral.sum[name] ?? 0) + value;
    if (++neutral.samples === 45) {
      for (const name in neutral.sum) neutral.rest[name] = Math.min(0.6, neutral.sum[name] / 45);
      console.log('\nneutral face captured (restart the bridge to recapture)');
    }
  }
  // Adaptive smoothing per channel: calm when still, fast when moving.
  for (const [name, value] of Object.entries(raw)) {
    const rest = name.startsWith('eyeBlink') || name.startsWith('eyeLook') ? 0 : neutral.rest[name] ?? 0;
    const target = Math.max(0, (value - rest) / (1 - rest));
    const previous = smoothed[name] ?? target;
    const alpha = clamp(0.3 + Math.abs(target - previous) * 3, 0.3, 0.9);
    smoothed[name] = previous + (target - previous) * alpha;
  }
  const b = smoothed;
  const g = (name) => b[name] ?? 0;
  // VRChat native eye tracking: closed amount and gaze (degrees).
  const blink = ((tracking.blinkLeft ?? g('eyeBlinkLeft')) + (tracking.blinkRight ?? g('eyeBlinkRight'))) / 2;
  send('/tracking/eye/EyesClosedAmount', clamp(blink));
  send('/tracking/eye/CenterPitchYaw', clamp(-eyes.y, -1, 1) * 18, clamp(eyes.x, -1, 1) * 28);
  // Eyes (Unified Expressions: X right +, Y up +; EyeLid 0 closed, 0.75 open, 1 wide).
  // The tracker's processed gaze (relative to the calibrated straight-ahead look) instead of the
  // raw look shapes, which read "looking down" whenever the camera sits below the face.
  eyes.x += (clamp(tracking.eyeLookX ?? 0, -1, 1) - eyes.x) * 0.35;
  eyes.y += (clamp(tracking.eyeLookY ?? 0, -1, 1) - eyes.y) * 0.35;
  const leftX = eyes.x;
  const rightX = eyes.x;
  const leftY = eyes.y;
  const rightY = eyes.y;
  expression('EyeLeftX', leftX);
  expression('EyeRightX', rightX);
  expression('EyeLeftY', leftY);
  expression('EyeRightY', rightY);
  expression('EyeY', (leftY + rightY) / 2);
  // Calibrated blinks from the tracker (raw eyeBlink sits around 0.2-0.4 with glasses, which
  // left the lids half closed). Fast to close, a touch slower to open.
  const lid = (blinkValue, wide) => clamp((1 - clamp(blinkValue)) * 0.75 + wide * 0.25);
  const wantLeft = lid(tracking.blinkLeft ?? g('eyeBlinkLeft'), g('eyeWideLeft'));
  const wantRight = lid(tracking.blinkRight ?? g('eyeBlinkRight'), g('eyeWideRight'));
  eyes.lidLeft += (wantLeft - eyes.lidLeft) * (wantLeft < eyes.lidLeft ? 0.8 : 0.45);
  eyes.lidRight += (wantRight - eyes.lidRight) * (wantRight < eyes.lidRight ? 0.8 : 0.45);
  expression('EyeLidLeft', eyes.lidLeft);
  expression('EyeLidRight', eyes.lidRight);
  expression('EyeLid', (eyes.lidLeft + eyes.lidRight) / 2);
  expression('EyeSquintLeft', clamp(g('eyeSquintLeft')));
  expression('EyeSquintRight', clamp(g('eyeSquintRight')));
  // Jaw and mouth.
  expression('JawOpen', clamp(g('jawOpen') * 1.2));
  expression('JawX', clamp(g('jawRight') - g('jawLeft'), -1, 1));
  expression('JawForward', clamp(g('jawForward')));
  expression('MouthClosed', clamp(g('mouthClose')));
  expression('MouthUpperUp', clamp((g('mouthUpperUpLeft') + g('mouthUpperUpRight')) / 2));
  expression('MouthLowerDown', clamp((g('mouthLowerDownLeft') + g('mouthLowerDownRight')) / 2));
  expression('MouthOpen', clamp((g('mouthLowerDownLeft') + g('mouthLowerDownRight') + g('mouthUpperUpLeft') + g('mouthUpperUpRight')) / 2));
  expression('MouthX', clamp(g('mouthRight') - g('mouthLeft'), -1, 1));
  expression('MouthRaiserLower', clamp(g('mouthShrugLower')));
  expression('SmileSadLeft', clamp(g('mouthSmileLeft') - g('mouthFrownLeft') * 1.5, -1, 1));
  expression('SmileSadRight', clamp(g('mouthSmileRight') - g('mouthFrownRight') * 1.5, -1, 1));
  expression('MouthSmileLeft', clamp(g('mouthSmileLeft')));
  expression('MouthSmileRight', clamp(g('mouthSmileRight')));
  expression('MouthSadLeft', clamp(g('mouthFrownLeft') * 1.5));
  expression('MouthSadRight', clamp(g('mouthFrownRight') * 1.5));
  expression('LipPucker', clamp(g('mouthPucker')));
  expression('LipFunnelUpper', clamp(g('mouthFunnel')));
  expression('LipFunnelLower', clamp(g('mouthFunnel')));
  expression('LipSuckUpper', clamp(g('mouthRollUpper')));
  expression('LipSuckLower', clamp(g('mouthRollLower')));
  // Cheeks: puff positive, suck negative.
  const puff = clamp(g('cheekPuff') * 2.4);
  expression('CheekPuffSuckLeft', clamp(g('cheekPuffLeft') || puff));
  expression('CheekPuffSuckRight', clamp(g('cheekPuffRight') || puff));
  expression('CheekSquint', clamp((g('cheekSquintLeft') + g('cheekSquintRight')) / 2));
  // Brows.
  expression('BrowInnerUp', clamp(g('browInnerUp')));
  expression('BrowOuterUp', clamp((g('browOuterUpLeft') + g('browOuterUpRight')) / 2));
  expression('BrowDown', clamp((g('browDownLeft') + g('browDownRight')) / 2));
  // Tongue (from the native tongue detector).
  if (tongue) {
    // The avatar gets the tongue as 3 bits, so detector noise made it flicker between steps.
    // Smooth it, switch it on/off with hysteresis, and only aim it while it is out.
    const raw = clamp(g('tongueOut'));
    tongueState.level += (raw - tongueState.level) * 0.35;
    if (!tongueState.out && tongueState.level > 0.45) tongueState.out = true;
    else if (tongueState.out && tongueState.level < 0.25) tongueState.out = false;
    const out = tongueState.out ? clamp((tongueState.level - 0.25) / 0.6) : 0;
    tongueState.x += ((tongueState.out ? clamp(g('tongueRight') - g('tongueLeft'), -1, 1) : 0) - tongueState.x) * 0.3;
    tongueState.y += ((tongueState.out ? clamp(g('tongueUp') - g('tongueDown'), -1, 1) : 0) - tongueState.y) * 0.3;
    expression('TongueOut', out);
    expression('TongueX', Math.abs(tongueState.x) < 0.15 ? 0 : tongueState.x);
    expression('TongueY', Math.abs(tongueState.y) < 0.15 ? 0 : tongueState.y);
  }
  // Flags the avatar uses to switch face tracking on (sent in the type it declares).
  flag('EyeTrackingActive', 1);
  flag('LipTrackingActive', true);
  flag('ExpressionTrackingActive', true);
  return true;
}

function stop() {
  clearInterval(timer);
  clearInterval(status);
  flag('LipTrackingActive', false);
  flag('ExpressionTrackingActive', false);
  flag('EyeTrackingActive', 0);
  setTimeout(() => { socket.close(); try { listener.close(); } catch { /* closed */ } try { rmSync(out); } catch { /* already gone */ } process.exit(0); }, 100);
}
process.on('SIGINT', () => { child.kill(); stop(); });
