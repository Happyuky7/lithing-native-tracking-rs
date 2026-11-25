//! Port of src/app/hand-pose.ts: finger articulation and gesture
//! recognition from MediaPipe's 21 hand landmarks, plus the One Euro
//! filter. Same math, same constants, no allocation per frame.
//!
//! Landmarks: 0 wrist; thumb 1 CMC, 2 MCP, 3 IP, 4 tip; index 5-8, middle
//! 9-12, ring 13-16, little 17-20 (MCP, PIP, DIP, tip).

use crate::types::{ARTICULATION_SIZE, HandGesture, Side};
use std::f64::consts::PI;

/// Anatomical limits of each articulation value (see hand-pose.ts for the layout).
pub const ARTICULATION_MIN: [f32; ARTICULATION_SIZE] = [-0.35, -0.35, -0.2, -0.3, -0.35, -0.05, -0.05, -0.3, -0.35, -0.05, -0.05, -0.25, -0.35, -0.05, -0.05, -0.3, -0.35, -0.05, -0.05, -0.45];
pub const ARTICULATION_MAX: [f32; ARTICULATION_SIZE] = [1.45, 1.2, 1.1, 1.4, 1.6, 1.85, 1.4, 0.4, 1.6, 1.85, 1.4, 0.25, 1.6, 1.85, 1.4, 0.25, 1.6, 1.85, 1.4, 0.3];

/// Per-hand values the gesture classifier reads, besides the articulation.
#[derive(Clone, Copy, Debug)]
pub struct HandFeatures {
    /// How extended each finger is, 0 (curled) to 1 (straight), thumb..little.
    pub extension: [f32; 5],
    pub index_reach: f32,
    /// Thumb tip to index tip distance, in palm widths.
    pub pinch: f32,
    /// Vertical component of the thumb's direction in body space.
    pub thumb_up: f32,
}

impl Default for HandFeatures {
    fn default() -> Self {
        Self { extension: [0.0; 5], index_reach: 0.0, pinch: 1.0, thumb_up: 0.0 }
    }
}

const THUMB_REST_JOINTS: [usize; 6] = [6, 7, 10, 11, 14, 15];

type V3 = [f64; 3];

fn sub(body: &[V3; 21], to: usize, from: usize) -> V3 {
    [body[to][0] - body[from][0], body[to][1] - body[from][1], body[to][2] - body[from][2]]
}
fn dist(body: &[V3; 21], a: usize, b: usize) -> f64 {
    length(&sub(body, a, b))
}
fn dot(a: &V3, b: &V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn length(a: &V3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
fn normalize(a: &mut V3) -> f64 {
    let size = length(a);
    if size > 1e-12 {
        a[0] /= size;
        a[1] /= size;
        a[2] /= size;
    }
    size
}
fn cross(a: &V3, b: &V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
/// Angle from `a` to `b` around unit `axis`, both projected onto the plane normal to it.
fn signed_angle(a: &V3, b: &V3, axis: &V3) -> f64 {
    let aa = dot(a, axis);
    let ba = dot(b, axis);
    let pa = [a[0] - axis[0] * aa, a[1] - axis[1] * aa, a[2] - axis[2] * aa];
    let pb = [b[0] - axis[0] * ba, b[1] - axis[1] * ba, b[2] - axis[2] * ba];
    let c = cross(&pa, &pb);
    (c[0] * axis[0] + c[1] * axis[1] + c[2] * axis[2]).atan2(dot(&pa, &pb))
}
fn unsigned_angle(a: &V3, b: &V3) -> f64 {
    let lengths = length(a) * length(b);
    if lengths < 1e-12 { 0.0 } else { (dot(a, b) / lengths).clamp(-1.0, 1.0).acos() }
}
fn clamp01(v: f64) -> f64 {
    v.clamp(0.0, 1.0)
}

/// Measures 21 landmarks (preferably metric world landmarks) into joint
/// angles `out` and `features`. `mirror_x` is the camera to body X sign; with
/// it the hand has the chirality of the avatar `side` it drives. Returns
/// false for a degenerate hand (leaving `out` untouched).
pub fn measure_hand(points: &[[f32; 3]; 21], mirror_x: f32, side: Side, out: &mut [f32; ARTICULATION_SIZE], features: &mut HandFeatures) -> bool {
    let mut body = [[0f64; 3]; 21];
    for (b, p) in body.iter_mut().zip(points) {
        *b = [p[0] as f64 * mirror_x as f64, -(p[1] as f64), -(p[2] as f64)];
    }
    let mut f = sub(&body, 9, 0);
    let mut a = sub(&body, 5, 17);
    let palm_width = length(&a);
    if normalize(&mut f) < 1e-6 || palm_width < 1e-6 {
        return false;
    }
    let along = dot(&a, &f);
    a = [a[0] - f[0] * along, a[1] - f[1] * along, a[2] - f[2] * along];
    if normalize(&mut a) < 1e-6 {
        return false;
    }
    let s = cross(&f, &a);
    let sign = if side == Side::Left { 1.0 } else { -1.0 };
    let n = [s[0] * sign, s[1] * sign, s[2] * sign];
    let k = cross(&f, &n);

    for finger in 0..4 {
        let base = 5 + finger * 4;
        let offset = 4 + finger * 4;
        let u = sub(&body, base + 1, base);
        let v = sub(&body, base + 2, base + 1);
        let w = sub(&body, base + 3, base + 2);
        let proximal = length(&u);
        let finger_length = proximal + length(&v) + length(&w);
        let mcp = signed_angle(&f, &u, &k);
        out[offset] = mcp as f32;
        out[offset + 1] = signed_angle(&u, &v, &k) as f32;
        out[offset + 2] = signed_angle(&v, &w, &k) as f32;
        let in_plane = if proximal > 1e-9 { (1.0 - (dot(&u, &s) / proximal).powi(2)).max(0.0).sqrt() } else { 0.0 };
        let spread_weight = clamp01((mcp.clamp(0.0, PI / 2.0).cos() - 0.25) / 0.6) * clamp01((in_plane - 0.3) / 0.5);
        out[offset + 3] = (signed_angle(&f, &u, &s) * spread_weight) as f32;
        let reach = if finger_length > 1e-9 { (dist(&body, base + 3, 0) - dist(&body, base, 0)) / finger_length } else { 0.0 };
        let reach_score = clamp01((reach - 0.15) / 0.65);
        let curl_score = 1.0 - clamp01((out[offset] as f64 + out[offset + 1] as f64 + out[offset + 2] as f64 - 0.6) / 1.9);
        features.extension[finger + 1] = (reach_score * 0.6 + curl_score * 0.4) as f32;
        if finger == 0 {
            features.index_reach = reach_score as f32;
        }
    }

    let mut u = sub(&body, 2, 1);
    let v = sub(&body, 3, 2);
    let w = sub(&body, 4, 3);
    if normalize(&mut u) > 1e-9 {
        out[0] = dot(&u, &a).atan2(dot(&u, &f)) as f32;
        out[1] = dot(&u, &n).clamp(-1.0, 1.0).asin() as f32;
    }
    out[2] = unsigned_angle(&u, &v) as f32;
    out[3] = unsigned_angle(&v, &w) as f32;
    let thumb_straight = 1.0 - clamp01((out[2] as f64 + out[3] as f64 - 0.45) / 1.0);
    let tucked = THUMB_REST_JOINTS.iter().map(|&j| dist(&body, 4, j)).fold(f64::INFINITY, f64::min);
    let thumb_clear = clamp01((tucked / palm_width - 0.35) / 0.45);
    let thumb_away = clamp01((dist(&body, 4, 17) / palm_width - 1.0) / 0.6);
    features.extension[0] = (thumb_clear * 0.55 + thumb_straight * 0.25 + thumb_away * 0.2) as f32;
    features.pinch = (dist(&body, 4, 8).min(dist(&body, 4, 7)) / palm_width) as f32;
    let mut t = sub(&body, 4, 2);
    features.thumb_up = if normalize(&mut t) > 1e-9 { t[1] as f32 } else { 0.0 };
    true
}

/// Clamps every value to its anatomical range.
pub fn clamp_articulation(values: &mut [f32; ARTICULATION_SIZE]) {
    for (i, v) in values.iter_mut().enumerate() {
        *v = v.clamp(ARTICULATION_MIN[i], ARTICULATION_MAX[i]);
    }
}

fn smoothing_alpha(delta: f64, cutoff: f64) -> f64 {
    let time_constant = 1.0 / (2.0 * PI * cutoff.max(0.01));
    1.0 / (1.0 + time_constant / delta)
}

/// One Euro filter over N values (Casiez et al. 2012), as in hand-pose.ts.
#[derive(Clone, Debug)]
pub struct OneEuroFilter<const N: usize> {
    value: [f32; N],
    derivative: [f32; N],
    last_time: f64,
}

impl<const N: usize> Default for OneEuroFilter<N> {
    fn default() -> Self {
        Self { value: [0.0; N], derivative: [0.0; N], last_time: f64::NEG_INFINITY }
    }
}

impl<const N: usize> OneEuroFilter<N> {
    pub fn reset(&mut self) {
        self.last_time = f64::NEG_INFINITY;
    }

    /// Filters `values` in place; `time` in milliseconds.
    pub fn filter(&mut self, values: &mut [f32; N], time: f64, min_cutoff: f64, beta: f64, derivative_cutoff: f64) {
        let delta = (time - self.last_time) / 1000.0;
        if !delta.is_finite() || delta <= 0.0 || delta > 0.6 {
            self.value = *values;
            self.derivative = [0.0; N];
            self.last_time = time;
            return;
        }
        self.last_time = time;
        let derivative_alpha = smoothing_alpha(delta, derivative_cutoff) as f32;
        for i in 0..N {
            let previous = self.value[i];
            self.derivative[i] += ((values[i] - previous) / delta as f32 - self.derivative[i]) * derivative_alpha;
            let cutoff = min_cutoff + beta * self.derivative[i].abs() as f64;
            self.value[i] = previous + (values[i] - previous) * smoothing_alpha(delta, cutoff) as f32;
            values[i] = self.value[i];
        }
    }
}

/// Finger states for each gesture: 1 extended, 0 curled, -1 either. Thumb..little.
const PATTERNS: [(HandGesture, [i8; 5]); 7] = [
    (HandGesture::Open, [-1, 1, 1, 1, 1]),
    (HandGesture::Victory, [-1, 1, 1, 0, 0]),
    (HandGesture::Pointing, [-1, 1, 0, 0, 0]),
    (HandGesture::LoveYou, [1, 1, 0, 0, 1]),
    (HandGesture::Rock, [0, 1, 0, 0, 1]),
    (HandGesture::CallMe, [1, 0, 0, 0, 1]),
    (HandGesture::Fist, [0, 0, 0, 0, 0]),
];

/// Recognises gestures with per-finger hysteresis and a hold time (see hand-pose.ts).
#[derive(Clone, Debug)]
pub struct GestureClassifier {
    pub gesture: HandGesture,
    pub confidence: f32,
    extended: [u8; 5],
    initialized: bool,
    pinching: bool,
    candidate: HandGesture,
    candidate_frames: u32,
    required_frames: u32,
}

impl Default for GestureClassifier {
    fn default() -> Self {
        Self { gesture: HandGesture::None, confidence: 0.0, extended: [0; 5], initialized: false, pinching: false, candidate: HandGesture::None, candidate_frames: 0, required_frames: 3 }
    }
}

fn matches(extended: &[u8; 5], pattern: &[i8; 5]) -> bool {
    (0..5).all(|f| pattern[f] < 0 || extended[f] as i8 == pattern[f])
}

fn agreement(extension: &[f32; 5], pattern: &[i8; 5]) -> f32 {
    let (mut weakest, mut sum, mut count) = (1.0f32, 0.0f32, 0);
    for f in 0..5 {
        if pattern[f] < 0 {
            continue;
        }
        let value = if pattern[f] == 1 { extension[f] } else { 1.0 - extension[f] };
        weakest = weakest.min(value);
        sum += value;
        count += 1;
    }
    if count > 0 { score(weakest * 0.5 + sum / count as f32 * 0.5) } else { 0.0 }
}

fn score(value: f32) -> f32 {
    ((value - 0.4) / 0.5).clamp(0.0, 1.0)
}

impl GestureClassifier {
    pub fn pending(&self) -> HandGesture {
        self.candidate
    }
    pub fn finger_states(&self) -> [u8; 5] {
        self.extended
    }

    /// Detections a new gesture must hold, from the hand tracking rate (about 0.1 s).
    pub fn set_rate(&mut self, fps: f32) {
        self.required_frames = ((fps * 0.1).round() as u32).clamp(2, 5);
    }

    pub fn reset(&mut self) {
        let required = self.required_frames;
        *self = Self { required_frames: required, ..Self::default() };
    }

    /// Feeds one detection; true when the reported gesture changed.
    pub fn update(&mut self, features: &HandFeatures) -> bool {
        let extension = &features.extension;
        for f in 0..5 {
            let value = extension[f];
            if !self.initialized {
                self.extended[f] = (value > 0.5) as u8;
            } else if self.extended[f] == 1 && value < 0.38 {
                self.extended[f] = 0;
            } else if self.extended[f] == 0 && value > 0.62 {
                self.extended[f] = 1;
            }
        }
        self.initialized = true;
        self.pinching = if self.pinching { features.pinch < 0.48 } else { features.pinch < 0.34 };
        let [thumb, index, middle, ring, little] = self.extended.map(|v| v == 1);
        let mut found = HandGesture::None;
        let mut confidence = 0.0;
        if self.pinching && middle && (ring || little) {
            found = HandGesture::Ok;
            confidence = score(1.0 - ((features.pinch - 0.1) / 0.5).clamp(0.0, 1.0)).min(agreement(extension, &[-1, -1, 1, 1, 1]));
        } else if self.pinching && !middle && !ring && !little && features.index_reach > 0.12 && features.index_reach < 0.8 {
            found = HandGesture::FingerHeart;
            confidence = score(1.0 - ((features.pinch - 0.1) / 0.5).clamp(0.0, 1.0)).min(agreement(extension, &[-1, -1, 0, 0, 0]));
        } else if thumb && !index && !middle && !ring && !little {
            if features.thumb_up.abs() > 0.6 {
                found = if features.thumb_up > 0.0 { HandGesture::ThumbsUp } else { HandGesture::ThumbsDown };
                confidence = agreement(extension, &[1, 0, 0, 0, 0]).min(score(((features.thumb_up.abs() - 0.45) / 0.4).clamp(0.0, 1.0)));
            }
        } else {
            for (gesture, pattern) in PATTERNS.iter() {
                if matches(&self.extended, pattern) {
                    found = *gesture;
                    confidence = agreement(extension, pattern);
                    break;
                }
            }
        }
        if found == self.candidate {
            self.candidate_frames += 1;
        } else {
            self.candidate = found;
            self.candidate_frames = 1;
        }
        if self.candidate == self.gesture {
            let target = if found == HandGesture::None { 0.0 } else { confidence };
            self.confidence += (target - self.confidence) * 0.5;
            return false;
        }
        if self.candidate_frames < self.required_frames {
            return false;
        }
        self.gesture = self.candidate;
        self.confidence = if found == HandGesture::None { 0.0 } else { confidence };
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat open right hand facing the camera, fingers up (camera space, y down).
    pub(crate) fn open_hand() -> [[f32; 3]; 21] {
        let mut p = [[0f32; 3]; 21];
        p[0] = [0.0, 0.0, 0.0];
        // Thumb out to the image left.
        for (i, (x, y)) in [(-0.025, -0.02), (-0.045, -0.035), (-0.06, -0.05), (-0.075, -0.065)].iter().enumerate() {
            p[1 + i] = [*x, *y, 0.0];
        }
        for finger in 0..4 {
            let x = -0.02 + finger as f32 * 0.015;
            for joint in 0..4 {
                p[5 + finger * 4 + joint] = [x, -0.08 - joint as f32 * 0.025, 0.0];
            }
        }
        p
    }

    #[test]
    fn open_hand_is_straight_and_classified_open() {
        let mut out = [0.0; ARTICULATION_SIZE];
        let mut features = HandFeatures::default();
        assert!(measure_hand(&open_hand(), 1.0, Side::Left, &mut out, &mut features));
        for finger in 0..4 {
            assert!(out[4 + finger * 4].abs() < 0.1, "mcp {finger}: {}", out[4 + finger * 4]);
            assert!(features.extension[finger + 1] > 0.8);
        }
        let mut classifier = GestureClassifier::default();
        for _ in 0..3 {
            classifier.update(&features);
        }
        assert_eq!(classifier.gesture, HandGesture::Open);
    }

    #[test]
    fn one_euro_holds_still_values() {
        let mut filter = OneEuroFilter::<1>::default();
        let mut v = [1.0];
        filter.filter(&mut v, 0.0, 1.0, 0.0, 1.2);
        let mut v = [2.0];
        filter.filter(&mut v, 33.0, 1.0, 0.0, 1.2);
        assert!(v[0] > 1.0 && v[0] < 1.5, "{}", v[0]);
    }
}
