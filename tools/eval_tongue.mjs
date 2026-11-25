// Scores the native tongue detector on the hand-labelled CC clips the web
// detector was tuned on (same labels as the web evaluation: ragazza at
// 0.5 s steps, keshav640 / pamela640 as negatives for their first 59 s).
//   lithing-track run <clip> --lockstep --tongue --no-hands --face-fps 1000 --out <clip>.jsonl
//   node tools/eval_tongue.mjs ragazza.jsonl:24 keshav640.jsonl:25 pamela640.jsonl:30
import fs from 'node:fs';
import path from 'node:path';

const range = (a, b) => Array.from({ length: b - a + 1 }, (_, i) => a + i);
const ragazzaYes = [...range(5, 11), ...range(21, 37), 39, ...range(42, 45), 51, ...range(58, 62), ...range(66, 82)];
const ragazzaNo = [...range(0, 4), 15, 18, 19, 38, 46, 53, 56, 64, 65];
const runs = {};
for (const arg of process.argv.slice(2)) {
  const cut = arg.lastIndexOf(':'); const file = arg.slice(0, cut), fps = arg.slice(cut + 1);
  const clip = path.basename(file).replace(/\.jsonl$/, '').replace(/^tongue_/, '');
  runs[clip] = fs.readFileSync(file, 'utf8').trim().split('\n').map((l) => JSON.parse(l)).map((r) => ({ t: r.frame / Number(fps), out: r.tongue?.out ?? null }));
}
const labels = [];
for (const i of ragazzaYes) labels.push({ clip: 'ragazza', t: i * 0.5, yes: true });
for (const i of ragazzaNo) labels.push({ clip: 'ragazza', t: i * 0.5, yes: false });
for (const clip of ['keshav640', 'pamela640']) for (let t = 0; t < 59; t += 1) labels.push({ clip, t, yes: false });
const scored = [];
for (const label of labels) {
  const records = runs[label.clip];
  if (!records) continue;
  const record = records.reduce((best, r) => (Math.abs(r.t - label.t) < Math.abs(best.t - label.t) ? r : best));
  if (record.out !== null) scored.push({ ...label, out: record.out });
}
const positives = scored.filter((s) => s.yes), negatives = scored.filter((s) => !s.yes);
let auc = 0;
for (const p of positives) for (const n of negatives) auc += p.out > n.out ? 1 : p.out === n.out ? 0.5 : 0;
auc /= positives.length * negatives.length;
console.log(`labelled with a face: ${scored.length} (${positives.length} tongue out, ${negatives.length} not); AUC ${auc.toFixed(3)}`);
for (const threshold of [0.3, 0.5, 0.7]) {
  const tp = positives.filter((s) => s.out >= threshold).length, fp = negatives.filter((s) => s.out >= threshold).length;
  console.log(`threshold ${threshold}: recall ${(tp / positives.length).toFixed(2)}, precision ${(tp / Math.max(1, tp + fp)).toFixed(2)}, false positives ${fp}/${negatives.length}`);
}
