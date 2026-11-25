// Decodes payloads recorded by tests/roundtrip.rs with guest-js/decode.ts and
// prints them as JSON, for the Rust test to compare with its own decoding.
// File format: repeated [u32 little-endian length][payload].
import fs from 'node:fs';
import { decodeFrame } from './decode.ts';

const bytes = fs.readFileSync(process.argv[2]);
const out = [];
for (let offset = 0; offset < bytes.length;) {
  const length = bytes.readUInt32LE(offset);
  offset += 4;
  const frame = decodeFrame(bytes.subarray(offset, offset + length));
  offset += length;
  out.push({ ...frame, hands: frame.hands.map((h) => ({ ...h, fingers: Array.from(h.fingers) })) });
}
process.stdout.write(JSON.stringify(out));
