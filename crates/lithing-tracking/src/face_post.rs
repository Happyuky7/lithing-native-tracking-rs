//! Port of the face half of src/app/face-tracker.ts: head pose, gaze,
//! eye openness (eyes.rs), the eye-side vote, blendshape side naming and
//! the stability filter. Input: one frame's landmarks and blendshapes.

use crate::eyes::{EyeCalibration, EyeOpenness, EyesDebug, measure_eye_geometry};
use crate::gaze::{GazeCalibration, GazeSample, map_gaze_axis};
use crate::types::{BLENDSHAPE_CHANNELS, BLENDSHAPE_NAMES, FaceDetection, TrackingData};
use std::f32::consts::PI;

/// Settings the web tracker exposes for the face.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, rename_all = "camelCase"))]
pub struct FaceSettings {
    pub mirror: bool,
    pub glasses_mode: bool,
    pub low_light: bool,
    pub high_light: bool,
    /// 0 follows raw values, 1 smooths the most (web default 0.72).
    pub stability: f32,
    /// Settings → Eye openness: 0..1, 0.5 default (see eyes::eye_curve).
    pub eye_openness: f32,
    /// The user's eye calibration; None adapts from defaults.
    pub eye_calibration: Option<EyeCalibration>,
    /// The user's pupil calibration (gaze.rs); None keeps the forward
    /// baseline's offset, exactly as before the calibration existed.
    pub gaze_calibration: Option<GazeCalibration>,
}

impl Default for FaceSettings {
    fn default() -> Self {
        Self { mirror: true, glasses_mode: false, low_light: false, high_light: false, stability: 0.72, eye_openness: 0.5, eye_calibration: None, gaze_calibration: None }
    }
}

#[derive(Clone, Copy, Default, Debug)]
struct Baseline {
    yaw: f32,
    pitch: f32,
    roll: f32,
    eye_x: f32,
    eye_y: f32,
    x: f32,
    y: f32,
    size: f32,
    ratio_left: f32,
    ratio_right: f32,
}

impl Baseline {
    fn add(&mut self, o: &Baseline) {
        self.yaw += o.yaw;
        self.pitch += o.pitch;
        self.roll += o.roll;
        self.eye_x += o.eye_x;
        self.eye_y += o.eye_y;
        self.x += o.x;
        self.y += o.y;
        self.size += o.size;
        self.ratio_left += o.ratio_left;
        self.ratio_right += o.ratio_right;
    }
}

/// Partner of each blendshape across the face (Left <-> Right), itself when unsided.
fn side_partner() -> [usize; 52] {
    let mut partner = [0usize; 52];
    for (i, name) in BLENDSHAPE_NAMES.iter().enumerate() {
        partner[i] = i;
        let other = if let Some(stem) = name.strip_suffix("Left") {
            format!("{stem}Right")
        } else if let Some(stem) = name.strip_suffix("Right") {
            format!("{stem}Left")
        } else {
            continue;
        };
        if let Some(j) = BLENDSHAPE_NAMES.iter().position(|n| *n == other) {
            partner[i] = j;
        }
    }
    partner
}

const B_EYE_BLINK_LEFT: usize = 9;
const B_EYE_BLINK_RIGHT: usize = 10;
const B_JAW_OPEN: usize = 25;

pub struct FacePost {
    pub settings: FaceSettings,
    calibration_frames: u32,
    calibration_samples: u32,
    calibration_sum: Baseline,
    baseline: Baseline,
    eye_side_vote: f32,
    eye_sides_swapped: bool,
    /// Aspect-corrected eye distance and eye centre of the last face (the hands measure against them).
    pub face_scale: f32,
    pub face_center: (f32, f32),
    filtered: [f32; 14],
    previous: [f32; 14],
    velocity: [f32; 14],
    last_filter_time: f64,
    last_face_seen: f64,
    pose_history: Vec<[f32; 14]>,
    partner: [usize; 52],
    /// Blendshapes of the last face, avatar-side names (see `TrackingFrame::blendshapes`).
    pub blendshapes: Option<[f32; BLENDSHAPE_CHANNELS]>,
    /// Eye closure from the blendshape, the lid gap and the iris (eyes.rs).
    eyes: EyeOpenness,
    /// The eye cues and levels of the last face (debug view, eye calibration).
    pub eyes_debug: Option<EyesDebug>,
    /// The raw gaze of the last face, before baseline and mirroring (debug view, pupil calibration).
    pub gaze_debug: Option<GazeSample>,
}

impl Default for FacePost {
    fn default() -> Self {
        Self::new(FaceSettings::default())
    }
}

fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    v.clamp(lo, hi)
}
fn soft_deadzone(value: f32, threshold: f32) -> f32 {
    let m = value.abs();
    if m <= threshold { 0.0 } else { value.signum() * (m - threshold) }
}
fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    (a.0 - b.0).hypot(a.1 - b.1)
}
fn low_pass_alpha(delta: f32, cutoff: f32) -> f32 {
    let time_constant = 1.0 / (2.0 * PI * cutoff.max(0.01));
    1.0 / (1.0 + time_constant / delta)
}
fn median3(a: f32, b: f32, c: f32) -> f32 {
    a + b + c - a.min(b).min(c) - a.max(b).max(c)
}
fn raw_yaw_estimate(nose_offset: (f32, f32), eye_axis: (f32, f32), eye_distance: f32) -> f32 {
    -(nose_offset.0 * eye_axis.0 + nose_offset.1 * eye_axis.1) / eye_distance * 1.3
}

/// eyeLookX / eyeLookY from the raw gaze (as in face-tracker.ts). Without a
/// pupil calibration: the offset from the forward baseline, unchanged from
/// before the calibration existed. With one: each side scaled by its
/// measured edge around the calibration's own centre (gaze.rs), which then
/// replaces the baseline's eye_x / eye_y. Mirroring and the deadzone stay.
fn eye_look(raw_x: f32, raw_y: f32, b: &Baseline, mirror: bool, gaze: Option<&GazeCalibration>) -> (f32, f32) {
    match gaze {
        None => (soft_deadzone(clamp((raw_x - b.eye_x) * if mirror { -1.0 } else { 1.0 }, -1.0, 1.0), 0.025), soft_deadzone(clamp(raw_y - b.eye_y, -1.0, 1.0), 0.025)),
        Some(c) => (
            soft_deadzone(clamp(map_gaze_axis(raw_x, c.center.x, c.left, c.right) * if mirror { -1.0 } else { 1.0 }, -1.0, 1.0), 0.025),
            soft_deadzone(clamp(map_gaze_axis(raw_y, c.center.y, c.up, c.down), -1.0, 1.0), 0.025),
        ),
    }
}

// TrackingData field order (see TrackingData::NAMES).
const HEAD_KEYS: [usize; 6] = [0, 1, 2, 3, 4, 5];

fn step_limit(key: usize, delta: f32, low: bool, high: bool) -> f32 {
    let light = if low { 0.52 } else if high { 0.68 } else { 1.0 };
    match key {
        0..=5 => (0.018 + delta * 2.8) * light,
        9 | 10 => (0.03 + delta * 4.2) * light,
        6 => 0.55,
        7 | 8 => 1.0,
        _ => 0.35,
    }
}

fn filter_tuning(key: usize, stability: f32) -> (f32, f32) {
    match key {
        7 | 8 => (4.5 - stability * 1.5, 1.2 - stability * 0.35),
        6 => (3.2 - stability * 1.25, 0.9 - stability * 0.3),
        11..=13 => (2.4 - stability, 0.7 - stability * 0.25),
        9 | 10 => (2.1 - stability * 1.2, 0.65 - stability * 0.28),
        _ => (2.6 - stability * 1.55, 0.8 - stability * 0.32),
    }
}

impl FacePost {
    pub fn new(settings: FaceSettings) -> Self {
        let mut post = Self {
            settings,
            calibration_frames: 0,
            calibration_samples: 0,
            calibration_sum: Baseline::default(),
            baseline: Baseline::default(),
            eye_side_vote: 0.0,
            eye_sides_swapped: false,
            face_scale: 0.0,
            face_center: (0.5, 0.4),
            filtered: [0.0; 14],
            previous: [0.0; 14],
            velocity: [0.0; 14],
            last_filter_time: f64::NEG_INFINITY,
            last_face_seen: f64::NEG_INFINITY,
            pose_history: Vec::with_capacity(3),
            partner: side_partner(),
            blendshapes: None,
            eyes: EyeOpenness::default(),
            eyes_debug: None,
            gaze_debug: None,
        };
        post.eyes.low_light = post.settings.low_light;
        post.eyes.set_calibration(post.settings.eye_calibration);
        post.calibrate();
        post
    }

    /// Starts a 24-frame calibration of the neutral pose (as on camera start).
    pub fn calibrate(&mut self) {
        self.calibration_frames = 24;
        self.calibration_samples = 0;
        self.calibration_sum = Baseline::default();
        self.filtered = [0.0; 14];
        self.previous = [0.0; 14];
        self.velocity = [0.0; 14];
        self.last_filter_time = f64::NEG_INFINITY;
        self.last_face_seen = f64::NEG_INFINITY;
        self.pose_history.clear();
    }

    pub fn calibrating(&self) -> bool {
        self.calibration_frames > 0
    }

    fn vote_eye_sides(&mut self, geometric: f32, blend: f32) {
        if geometric.abs() < 0.4 || blend.abs() < 0.25 {
            return;
        }
        let agrees = geometric.signum() == blend.signum();
        self.eye_side_vote = clamp(self.eye_side_vote + if agrees { 0.08 } else { -0.08 }, -1.0, 1.0);
        if if self.eye_sides_swapped { self.eye_side_vote > 0.3 } else { self.eye_side_vote < -0.3 } {
            self.eye_sides_swapped = !self.eye_sides_swapped;
        }
    }

    /// One face frame at `time` (ms) of a `width`x`height` camera; returns the smoothed TrackingData.
    pub fn update(&mut self, face: &FaceDetection, width: u32, height: u32, time: f64) -> TrackingData {
        self.last_face_seen = time;
        let points = &face.landmarks;
        let p = |i: usize| (points[i][0], points[i][1]);
        let avg = |idx: &[usize]| {
            let (mut x, mut y) = (0.0, 0.0);
            for &i in idx {
                x += points[i][0];
                y += points[i][1];
            }
            (x / idx.len() as f32, y / idx.len() as f32)
        };
        let aspect = if width > 0 { height as f32 / width as f32 } else { 0.75 };
        let left_eye = avg(&[33, 133, 159, 145]);
        let right_eye = avg(&[263, 362, 386, 374]);
        let nose = avg(&[1, 4, 5, 195]);
        let eye_center = ((left_eye.0 + right_eye.0) / 2.0, (left_eye.1 + right_eye.1) / 2.0);
        let eye_delta = (right_eye.0 - left_eye.0, right_eye.1 - left_eye.1);
        let eye_distance = eye_delta.0.hypot(eye_delta.1).max(0.001);
        let eye_axis = (eye_delta.0 / eye_distance, eye_delta.1 / eye_distance);
        self.face_scale = eye_delta.0.hypot(eye_delta.1 * aspect);
        self.face_center = eye_center;
        let face_down = (-eye_axis.1, eye_axis.0);
        let nose_offset = (nose.0 - eye_center.0, nose.1 - eye_center.1);
        let scores = &face.blendshapes;
        let eye_ratio = |outer: usize, inner: usize, ua: usize, ub: usize, la: usize, lb: usize| {
            let w = dist(p(outer), p(inner)).max(0.001);
            (dist(p(ua), p(la)) + dist(p(ub), p(lb))) / (2.0 * w)
        };
        let ratio_right = eye_ratio(33, 133, 159, 158, 145, 153);
        let ratio_left = eye_ratio(362, 263, 386, 385, 374, 380);
        let geometric_closure = |ratio: f32, open: f32| if open > 0.05 { clamp((open * 0.84 - ratio) / (open * 0.56), 0.0, 1.0) } else { clamp((0.18 - ratio) / 0.11, 0.0, 1.0) };
        let geo_right = geometric_closure(ratio_right, self.baseline.ratio_right);
        let geo_left = geometric_closure(ratio_left, self.baseline.ratio_left);
        self.vote_eye_sides(geo_right - geo_left, scores[B_EYE_BLINK_RIGHT] - scores[B_EYE_BLINK_LEFT]);
        let swapped = self.eye_sides_swapped;
        let partner = self.partner;
        // A model score by its MediaPipe name, sides corrected by the eye vote.
        let eye_blend = |index: usize| if swapped { scores[partner[index]] } else { scores[index] };
        let idx = |name: &str| BLENDSHAPE_NAMES.iter().position(|n| *n == name).unwrap();
        // Eye closure from the blendshape, the lid gap and the iris together (eyes.rs).
        if self.eyes.calibration() != self.settings.eye_calibration.as_ref() {
            self.eyes.set_calibration(self.settings.eye_calibration);
        }
        self.eyes.sensitivity = self.settings.eye_openness;
        self.eyes.glasses = self.settings.glasses_mode;
        self.eyes.low_light = self.settings.low_light;
        let mut cues_right = measure_eye_geometry(points, 0, aspect);
        let mut cues_left = measure_eye_geometry(points, 1, aspect);
        cues_right.blink = eye_blend(B_EYE_BLINK_RIGHT);
        cues_left.blink = eye_blend(B_EYE_BLINK_LEFT);
        // Each eye on its own: linking near-equal eyes cost accuracy on one-sided squints and smiles.
        let (blink_right, blink_left) = self.eyes.update(cues_right, cues_left, time);
        self.eyes_debug = Some(self.eyes.debug);
        let iris_offset = |iris: usize, ca: usize, cb: usize, upper: usize, lower: usize| {
            if points.len() <= iris {
                return (0.0, 0.0);
            }
            let horizontal = (p(cb).0 - p(ca).0).abs().max(0.001);
            let vertical = (p(lower).1 - p(upper).1).abs().max(0.001);
            ((p(iris).0 - (p(ca).0 + p(cb).0) / 2.0) / horizontal * 2.0, (p(iris).1 - (p(upper).1 + p(lower).1) / 2.0) / vertical * 2.0)
        };
        let gaze_right = (1.0 - blink_right).max(0.05);
        let gaze_left = (1.0 - blink_left).max(0.05);
        let gaze_mix = |r: f32, l: f32| (r * gaze_right + l * gaze_left) / (gaze_right + gaze_left);
        let iris_right = iris_offset(468, 33, 133, 159, 145);
        let iris_left = iris_offset(473, 362, 263, 386, 374);
        let iris_x = clamp(gaze_mix(iris_right.0, iris_left.0), -1.0, 1.0);
        let iris_y = clamp(gaze_mix(iris_right.1, iris_left.1), -1.0, 1.0);
        let blend_eye_x = gaze_mix(eye_blend(idx("eyeLookInRight")) - eye_blend(idx("eyeLookOutRight")), eye_blend(idx("eyeLookOutLeft")) - eye_blend(idx("eyeLookInLeft")));
        let blend_eye_y = gaze_mix(eye_blend(idx("eyeLookUpRight")) - eye_blend(idx("eyeLookDownRight")), eye_blend(idx("eyeLookUpLeft")) - eye_blend(idx("eyeLookDownLeft")));
        let glasses = self.settings.glasses_mode;
        let yaw_estimate = raw_yaw_estimate(nose_offset, eye_axis, eye_distance);
        let raw = Baseline {
            yaw: clamp(yaw_estimate, -0.7, 0.7),
            pitch: clamp(-(nose_offset.0 * face_down.0 + nose_offset.1 * face_down.1) / eye_distance * 1.1, -0.5, 0.5),
            roll: eye_delta.1.atan2(eye_delta.0),
            eye_x: blend_eye_x * if glasses { 0.8 } else { 0.55 } + iris_x * if glasses { 0.2 } else { 0.45 },
            eye_y: blend_eye_y * if glasses { 0.85 } else { 0.65 } - iris_y * if glasses { 0.15 } else { 0.35 },
            x: eye_center.0,
            y: eye_center.1,
            size: eye_distance / clamp(yaw_estimate, -0.8, 0.8).asin().cos().max(0.6),
            ratio_left,
            ratio_right,
        };
        self.gaze_debug = Some(GazeSample { x: raw.eye_x, y: raw.eye_y });
        if self.calibration_frames > 0 {
            self.calibration_sum.add(&raw);
            self.calibration_samples += 1;
            self.calibration_frames -= 1;
            if self.calibration_frames == 0 && self.calibration_samples > 0 {
                let n = self.calibration_samples as f32;
                let s = self.calibration_sum;
                self.baseline = Baseline {
                    yaw: s.yaw / n,
                    pitch: s.pitch / n,
                    roll: s.roll / n,
                    eye_x: s.eye_x / n,
                    eye_y: s.eye_y / n,
                    x: s.x / n,
                    y: s.y / n,
                    size: s.size / n,
                    ratio_right: s.ratio_right / n,
                    ratio_left: s.ratio_left / n,
                };
            }
        }
        let calibrating = self.calibration_frames > 0;
        let b = self.baseline;
        let mirror = self.settings.mirror;
        let blend = |name: &str| scores[idx(name)];
        let jaw = scores[B_JAW_OPEN];
        let (eye_look_x, eye_look_y) = if calibrating { (0.0, 0.0) } else { eye_look(raw.eye_x, raw.eye_y, &b, mirror, self.settings.gaze_calibration.as_ref()) };
        let measured = TrackingData {
            head_yaw: if calibrating { 0.0 } else { soft_deadzone(clamp(raw.yaw - b.yaw, -0.7, 0.7), 0.006) },
            head_pitch: if calibrating { 0.0 } else { soft_deadzone(clamp(raw.pitch - b.pitch, -0.5, 0.5), 0.006) },
            head_roll: if calibrating { 0.0 } else { soft_deadzone(clamp(raw.roll - b.roll, -0.5, 0.5), 0.008) },
            head_x: if calibrating { 0.0 } else { soft_deadzone(clamp(-(raw.x - b.x) / 0.22, -1.0, 1.0), 0.02) },
            head_y: if calibrating { 0.0 } else { soft_deadzone(clamp(-(raw.y - b.y) / 0.2, -1.0, 1.0), 0.02) },
            head_z: if calibrating || b.size <= 0.0 { 0.0 } else { soft_deadzone(clamp((raw.size / b.size - 1.0) / 0.35, -1.0, 1.0), 0.03) },
            mouth_open: clamp(if jaw != 0.0 { jaw } else { dist(p(13), p(14)) / eye_distance * 3.0 }, 0.0, 1.0),
            blink_left: if mirror { blink_right } else { blink_left },
            blink_right: if mirror { blink_left } else { blink_right },
            eye_look_x,
            eye_look_y,
            happy: clamp((blend("mouthSmileLeft") + blend("mouthSmileRight")) / 2.0, 0.0, 1.0),
            angry: clamp((blend("browDownLeft") + blend("browDownRight")) / 2.0, 0.0, 1.0),
            surprised: clamp(blend("eyeWideLeft").max(blend("eyeWideRight")).max(jaw), 0.0, 1.0),
        };
        self.capture_blendshapes(scores);
        self.smooth(measured.to_array(), time)
    }

    /// Avatar-side naming: MediaPipe's sides, corrected by the eye vote, swapped again when mirrored.
    fn capture_blendshapes(&mut self, scores: &[f32; 52]) {
        let swap = self.eye_sides_swapped != self.settings.mirror;
        let mut out = [0.0f32; BLENDSHAPE_CHANNELS];
        for (i, &score) in scores.iter().enumerate() {
            out[if swap { self.partner[i] } else { i }] = score;
        }
        // tongueOut starts as MediaPipe's (it has none: 0); the tongue detector adds to it.
        self.blendshapes = Some(out);
    }

    /// A frame without a face: hold the last pose briefly, then ease to neutral.
    pub fn missing(&mut self, time: f64) -> TrackingData {
        let hold = if self.settings.low_light { 950.0 } else if self.settings.high_light { 550.0 } else { 350.0 };
        if time - self.last_face_seen >= hold {
            self.blendshapes = None;
            self.eyes_debug = None;
            self.gaze_debug = None;
        }
        if time - self.last_face_seen < hold {
            let held = self.filtered;
            return self.smooth(held, time);
        }
        self.smooth([0.0; 14], time)
    }

    pub fn face_lost_for(&self, time: f64) -> f64 {
        time - self.last_face_seen
    }

    fn smooth(&mut self, input: [f32; 14], time: f64) -> TrackingData {
        if !self.last_filter_time.is_finite() || time - self.last_filter_time > 500.0 {
            self.filtered = input;
            self.previous = input;
            self.velocity = [0.0; 14];
            self.last_filter_time = time;
            return TrackingData::from_array(self.filtered);
        }
        let delta = clamp(((time - self.last_filter_time) / 1000.0) as f32, 1.0 / 240.0, 0.2);
        self.last_filter_time = time;
        let (low, high) = (self.settings.low_light, self.settings.high_light);
        let stability = clamp(self.settings.stability + if low { 0.24 } else if high { 0.14 } else { 0.0 }, 0.0, 1.0);
        if self.pose_history.len() == 3 {
            self.pose_history.remove(0);
        }
        self.pose_history.push(input);
        let mut source = input;
        if (low || high) && self.pose_history.len() == 3 {
            for key in HEAD_KEYS.iter().copied().chain([9, 10]) {
                source[key] = median3(self.pose_history[0][key], self.pose_history[1][key], self.pose_history[2][key]);
            }
        }
        for key in 0..14 {
            let max_step = step_limit(key, delta, low, high);
            let target = self.filtered[key] + clamp(source[key] - self.filtered[key], -max_step, max_step);
            let derivative = (target - self.previous[key]) / delta;
            self.velocity[key] += (derivative - self.velocity[key]) * low_pass_alpha(delta, 1.25);
            let (min_cutoff, beta) = filter_tuning(key, stability);
            let cutoff = min_cutoff + beta * self.velocity[key].abs();
            self.filtered[key] += (target - self.filtered[key]) * low_pass_alpha(delta, cutoff);
            self.previous[key] = target;
        }
        TrackingData::from_array(self.filtered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partners_pair_sides() {
        let partner = side_partner();
        assert_eq!(BLENDSHAPE_NAMES[partner[B_EYE_BLINK_LEFT]], "eyeBlinkRight");
        assert_eq!(partner[B_JAW_OPEN], B_JAW_OPEN);
    }

    /// Without a pupil calibration the gaze is exactly the pre-calibration formula.
    #[test]
    fn no_gaze_calibration_keeps_the_old_formula() {
        let old = |raw_x: f32, raw_y: f32, b: &Baseline, mirror: bool| {
            (soft_deadzone(clamp((raw_x - b.eye_x) * if mirror { -1.0 } else { 1.0 }, -1.0, 1.0), 0.025), soft_deadzone(clamp(raw_y - b.eye_y, -1.0, 1.0), 0.025))
        };
        let baselines = [Baseline::default(), Baseline { eye_x: 0.07, eye_y: -0.12, ..Default::default() }, Baseline { eye_x: -0.3, eye_y: 0.4, ..Default::default() }];
        let raws = [(0.0, 0.0), (0.01, -0.02), (0.2, -0.15), (-0.45, 0.3), (1.6, -1.9), (0.08, -0.11)];
        for b in &baselines {
            for &(x, y) in &raws {
                for mirror in [true, false] {
                    let (nx, ny) = eye_look(x, y, b, mirror, None);
                    let (ox, oy) = old(x, y, b, mirror);
                    assert_eq!(nx.to_bits(), ox.to_bits());
                    assert_eq!(ny.to_bits(), oy.to_bits());
                }
            }
        }
    }

    #[test]
    fn gaze_calibration_replaces_the_baseline_centre() {
        use crate::gaze::GazeCenter;
        let c = GazeCalibration { center: GazeCenter { x: 0.1, y: -0.05 }, left: -0.1, right: 0.3, up: 0.15, down: -0.25, captured_at: 0.0 };
        // The baseline's eye_x / eye_y no longer matter.
        let b = Baseline { eye_x: 0.5, eye_y: 0.5, ..Default::default() };
        assert_eq!(eye_look(0.1, -0.05, &b, true, Some(&c)), (0.0, 0.0));
        // Looking at the right edge: full range, mirrored.
        let (x, y) = eye_look(0.3, -0.05, &b, true, Some(&c));
        assert!((x - (-0.975)).abs() < 1e-6 && y == 0.0);
        let (x, _) = eye_look(0.3, -0.05, &b, false, Some(&c));
        assert!((x - 0.975).abs() < 1e-6);
        // Half way to the bottom edge.
        let (_, y) = eye_look(0.1, -0.15, &b, false, Some(&c));
        assert!((y - (-0.475)).abs() < 1e-6);
    }
}
