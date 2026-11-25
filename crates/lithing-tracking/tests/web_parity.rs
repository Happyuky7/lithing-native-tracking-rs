//! The hand-pose port against the web original: the same landmarks through
//! src/app/hand-pose.ts (run by Node) and through hand_pose.rs must give the
//! same joint angles, features, filtered values and gestures.
//!
//! Needs Node 22.6+ and the web app's source: this folder lives inside the
//! app's repository (../../../src/app/hand-pose.ts), or set LITHING_WEB_REPO.
//! Skipped (with a note) when either is missing.

use lithing_tracking::hand_pose::{GestureClassifier, HandFeatures, OneEuroFilter, clamp_articulation, measure_hand};
use lithing_tracking::{ARTICULATION_SIZE, Side};
use std::path::PathBuf;

/// Small deterministic generator (xorshift), so the cases are the same every run.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
}

/// A plausible hand (metres, camera space) with random curls, spread and orientation.
fn random_hand(rng: &mut Rng, pattern: [bool; 5], toward: f32) -> [[f32; 3]; 21] {
    let mut p = [[0f32; 3]; 21];
    // Fingers in the palm plane (x across, y up the hand), curling towards -z.
    let bases = [(-0.022, -0.085), (0.0, -0.09), (0.018, -0.085), (0.034, -0.075)];
    let lengths = [[0.04, 0.025, 0.02], [0.045, 0.028, 0.022], [0.042, 0.026, 0.021], [0.032, 0.02, 0.018]];
    for f in 0..4 {
        let curl = if pattern[f + 1] { rng.range(-0.05, 0.1) } else { rng.range(0.9, 1.3) };
        let spread = rng.range(-0.25, 0.25);
        let (mut x, mut y, mut z) = (bases[f].0, bases[f].1, 0.0f32);
        p[5 + f * 4] = [x, y, z];
        let mut angle = 0.0f32;
        for j in 0..3 {
            angle += curl * rng.range(0.8, 1.4);
            let l = lengths[f][j];
            x += l * spread.sin();
            y -= l * angle.cos() * spread.cos();
            z -= toward * l * angle.sin();
            p[6 + f * 4 + j] = [x, y, z];
        }
    }
    let thumb_out = if pattern[0] { rng.range(0.8, 1.0) } else { rng.range(0.0, 0.2) };
    let mut t = [-0.02f32, -0.025, 0.0];
    for j in 0..4 {
        p[1 + j] = t;
        let a = 0.6 + thumb_out * 0.6 + j as f32 * (1.0 - thumb_out) * 0.5;
        t = [t[0] - 0.025 * a.sin(), t[1] - 0.02 * a.cos(), t[2] - toward * 0.012 * (1.0 - thumb_out)];
    }
    // Random rotation (yaw, pitch, roll) and noise.
    let (a, b, c) = (rng.range(-1.2, 1.2), rng.range(-0.8, 0.8), rng.range(-3.0, 3.0));
    for q in p.iter_mut() {
        let (x, y, z) = (q[0], q[1], q[2]);
        let (x, y) = (x * c.cos() - y * c.sin(), x * c.sin() + y * c.cos());
        let (y, z) = (y * b.cos() - z * b.sin(), y * b.sin() + z * b.cos());
        let (x, z) = (x * a.cos() + z * a.sin(), -x * a.sin() + z * a.cos());
        *q = [x + rng.range(-0.002, 0.002), y + rng.range(-0.002, 0.002), z + rng.range(-0.002, 0.002)];
    }
    p
}

#[test]
fn hand_pose_matches_web() {
    let root = std::env::var_os("LITHING_WEB_REPO").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."));
    let module = root.join("src/app/hand-pose.ts");
    if !module.is_file() {
        eprintln!("skipped: {} not found", module.display());
        return;
    }
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let mut cases = Vec::new();
    let mut time = 0.0f64;
    for i in 0..400 {
        // Runs of one finger pattern (thumb..little extended?) so the classifier's hold time is exercised.
        let patterns = [[true; 5], [false; 5], [false, true, true, false, false], [false, true, false, false, false], [true, false, false, false, false], [true, true, false, false, true], [false, true, false, false, true], [true, false, false, false, true]];
        let pattern = patterns[(i / 12) % patterns.len()];
        let toward = if (i / 96) % 2 == 0 { 1.0 } else { -1.0 };
        let side = if (i / 50) % 2 == 0 { Side::Left } else { Side::Right };
        let mirror = if (i / 100) % 2 == 0 { 1.0f32 } else { -1.0 };
        time += 33.0;
        cases.push((random_hand(&mut rng, pattern, toward), side, mirror, time));
    }
    let json: Vec<serde_json::Value> = cases.iter().map(|(points, side, mirror, time)| serde_json::json!({ "points": points, "side": side.name(), "mirror": mirror, "time": time })).collect();
    let input = std::env::temp_dir().join(format!("lithing-hand-cases-{}.json", std::process::id()));
    std::fs::write(&input, serde_json::to_string(&json).unwrap()).unwrap();
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/hand_pose_parity.mjs");
    let output = match std::process::Command::new("node").arg(&script).arg(&module).arg(&input).output() {
        Ok(o) if o.status.success() => o,
        Ok(o) => panic!("node failed: {}", String::from_utf8_lossy(&o.stderr)),
        Err(_) => {
            eprintln!("skipped: node not found");
            return;
        }
    };
    let web: Vec<serde_json::Value> = serde_json::from_slice(&output.stdout).unwrap();
    let mut classifiers = [GestureClassifier::default(), GestureClassifier::default()];
    let mut filters = [OneEuroFilter::<ARTICULATION_SIZE>::default(), OneEuroFilter::<ARTICULATION_SIZE>::default()];
    let (mut worst, mut gestures_seen, mut gesture_mismatches) = (0f32, std::collections::HashSet::new(), 0);
    for ((points, side, mirror, time), web) in cases.iter().zip(&web) {
        let mut out = [0f32; ARTICULATION_SIZE];
        let mut features = HandFeatures::default();
        let ok = measure_hand(points, *mirror, *side, &mut out, &mut features);
        assert_eq!(ok, web["ok"].as_bool().unwrap());
        let close = |a: f32, b: &serde_json::Value| (a - b.as_f64().unwrap() as f32).abs();
        for i in 0..ARTICULATION_SIZE {
            worst = worst.max(close(out[i], &web["raw"][i]));
        }
        clamp_articulation(&mut out);
        filters[side.index()].filter(&mut out, *time, 1.2, 0.55, 1.2);
        for i in 0..ARTICULATION_SIZE {
            worst = worst.max(close(out[i], &web["filtered"][i]));
        }
        for i in 0..5 {
            worst = worst.max(close(features.extension[i], &web["extension"][i]));
        }
        worst = worst.max(close(features.pinch, &web["pinch"])).max(close(features.thumb_up, &web["thumbUp"]));
        let classifier = &mut classifiers[side.index()];
        classifier.update(&features);
        gestures_seen.insert(classifier.gesture.name());
        if classifier.gesture.name() != web["gesture"].as_str().unwrap() {
            gesture_mismatches += 1;
        }
    }
    eprintln!("{} hands: worst difference {worst:.2e}, gestures {gestures_seen:?}, mismatches {gesture_mismatches}", cases.len());
    // f32 storage on both sides; f64 maths in both; small differences come from operation order.
    assert!(worst < 1e-4, "worst difference {worst}");
    assert_eq!(gesture_mismatches, 0);
    assert!(gestures_seen.len() >= 3, "the cases should exercise several gestures: {gestures_seen:?}");
}
