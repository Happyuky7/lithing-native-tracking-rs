// Runs the web app's src/app/hand-pose.ts on the hands in a JSON file and prints
// the results, for crates/lithing-tracking/tests/web_parity.rs to compare with the port.
//   node tools/hand_pose_parity.mjs <hand-pose.ts> <cases.json>
import fs from 'node:fs';
import { pathToFileURL } from 'node:url';

const [modulePath, casesPath] = process.argv.slice(2);
const hp = await import(pathToFileURL(modulePath).href);
const cases = JSON.parse(fs.readFileSync(casesPath, 'utf8'));
const out = [];
const classifiers = { left: new hp.GestureClassifier(), right: new hp.GestureClassifier() };
const filters = { left: new hp.OneEuroFilter(hp.ARTICULATION_SIZE), right: new hp.OneEuroFilter(hp.ARTICULATION_SIZE) };
for (const c of cases) {
  const points = c.points.map(([x, y, z]) => ({ x, y, z }));
  const articulation = new Float32Array(hp.ARTICULATION_SIZE);
  const features = hp.createHandFeatures();
  const ok = hp.measureHand(points, c.mirror, c.side, articulation, features);
  const raw = Array.from(articulation);
  hp.clampArticulation(articulation);
  const clamped = Array.from(articulation);
  filters[c.side].filter(articulation, c.time, 1.2, 0.55);
  classifiers[c.side].update(features);
  out.push({ ok, raw, clamped, filtered: Array.from(articulation), extension: Array.from(features.extension), indexReach: features.indexReach, pinch: features.pinch, thumbUp: features.thumbUp, gesture: classifiers[c.side].gesture, confidence: classifiers[c.side].confidence });
}
process.stdout.write(JSON.stringify(out));
