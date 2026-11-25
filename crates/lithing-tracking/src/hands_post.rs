//! Port of the hand half of src/app/face-tracker.ts: placement relative to
//! the face, side assignment that keeps crossed hands crossed, finger
//! articulation and gestures, One Euro smoothing and the hold time.

use crate::hand_pose::{self, GestureClassifier, HandFeatures, OneEuroFilter};
use crate::types::{ARTICULATION_SIZE, HandDetection, HandGesture, HandPose, Side};

/// The web app's hand tracking trade-off.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum HandQuality {
    Performance,
    #[default]
    Balanced,
    Precise,
}

impl HandQuality {
    fn wrist_tuning(self) -> (f64, f64) {
        match self {
            Self::Performance => (0.5, 1.5),
            Self::Balanced => (0.7, 2.0),
            Self::Precise => (1.0, 3.0),
        }
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, rename_all = "camelCase"))]
pub struct HandSettings {
    pub mirror: bool,
    pub quality: HandQuality,
    /// 0 follows the fingers raw, 1 smooths the most (web default 0.5).
    pub finger_smoothing: f32,
    pub low_light: bool,
    pub high_light: bool,
}

impl Default for HandSettings {
    fn default() -> Self {
        Self { mirror: true, quality: HandQuality::Balanced, finger_smoothing: 0.5, low_light: false, high_light: false }
    }
}

const TRACK_MEMORY_MS: f64 = 600.0;
const TRACK_MAX_DISTANCE: f32 = 4.0;
const TRACK_STALE_COST: f32 = 0.5;
const TRACK_STALE_MAX: f32 = 0.6;
const FRESH_TRACK_COST: f32 = 1.1;
const LABEL_WEIGHT: f32 = 1.2;
const TRACKED_LABEL_WEIGHT: f32 = 0.3;

#[derive(Clone, Copy, Debug)]
struct HandTrack {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    time: f64,
}

impl Default for HandTrack {
    fn default() -> Self {
        Self { x: 0.0, y: 0.0, vx: 0.0, vy: 0.0, time: f64::NEG_INFINITY }
    }
}

#[derive(Default, Clone)]
struct SideState {
    fingers: [f32; ARTICULATION_SIZE],
    filter: OneEuroFilter<ARTICULATION_SIZE>,
    classifier: GestureClassifier,
    features: HandFeatures,
    face: OneEuroFilter<2>,
    wrist: OneEuroFilter<2>,
    depth: OneEuroFilter<1>,
    orientation: OneEuroFilter<6>,
    track: HandTrack,
    last_seen: f64,
    smoothed: Option<HandPose>,
}

impl SideState {
    fn new() -> Self {
        Self { last_seen: f64::NEG_INFINITY, ..Default::default() }
    }
    fn reset_filters(&mut self) {
        self.filter.reset();
        self.classifier.reset();
        self.face.reset();
        self.wrist.reset();
        self.depth.reset();
        self.orientation.reset();
    }
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if length < 1e-9 { [0.0, 1.0, 0.0] } else { [v[0] / length, v[1] / length, v[2] / length] }
}

pub struct HandsPost {
    pub settings: HandSettings,
    sides: [SideState; 2],
    label_scores: [f32; 2],
    label_trust: f32,
    /// The hands getHands() reports: left then right, each held briefly after it was last seen.
    pub visible: Vec<HandPose>,
}

impl Default for HandsPost {
    fn default() -> Self {
        Self::new(HandSettings::default())
    }
}

impl HandsPost {
    pub fn new(settings: HandSettings) -> Self {
        Self { settings, sides: [SideState::new(), SideState::new()], label_scores: [0.0; 2], label_trust: 0.0, visible: Vec::with_capacity(2) }
    }

    /// Gestures must hold about 0.1 s at this hand detection rate.
    pub fn set_rate(&mut self, fps: f32) {
        for side in &mut self.sides {
            side.classifier.set_rate(fps);
        }
    }

    pub fn reset(&mut self) {
        for side in &mut self.sides {
            side.reset_filters();
            side.smoothed = None;
            side.last_seen = f64::NEG_INFINITY;
            side.track = HandTrack::default();
        }
        self.visible.clear();
    }

    /// Turns one hand detection at `time` (ms) into the tracked hands.
    /// `face_scale`/`face_center` come from the face of the same frame.
    pub fn process(&mut self, detections: &[HandDetection], width: u32, height: u32, face_scale: f32, face_center: (f32, f32), time: f64) {
        let aspect = if width > 0 { height as f32 / width as f32 } else { 0.75 };
        let mirror = self.settings.mirror;
        let mut hands: Vec<HandPose> = Vec::with_capacity(2);
        for (index, detection) in detections.iter().take(2).enumerate() {
            let landmarks = &detection.landmarks;
            let wrist = landmarks[0];
            let middle = landmarks[9];
            // MediaPipe names hands for a mirrored image; a first guess that assign_sides() decides.
            let handedness = if mirror { detection.label.other() } else { detection.label };
            self.label_scores[index] = detection.label_score;
            let world = &detection.world;
            let segment = |from: usize, to: usize| [world[to][0] - world[from][0], world[to][1] - world[from][1], world[to][2] - world[from][2]];
            let mirror_x = if mirror { 1.0 } else { -1.0 };
            let to_body = |v: [f32; 3]| normalize3([v[0] * mirror_x, -v[1], -v[2]]);
            let palm_length = (middle[0] - wrist[0]).hypot((middle[1] - wrist[1]) * aspect);
            let palm_width = (landmarks[5][0] - landmarks[17][0]).hypot((landmarks[5][1] - landmarks[17][1]) * aspect);
            let apparent = palm_length * 0.58 + palm_width * 0.42;
            let wrist_depth = if face_scale > 0.0 { ((apparent / face_scale / 1.4 - 1.15) / 0.85).clamp(-0.55, 1.0) } else { ((apparent - 0.095) / 0.115).clamp(-0.55, 1.0) };
            let face_unit = if face_scale > 0.0 { face_scale } else { 0.09 };
            let face_x = (wrist[0] - face_center.0) / face_unit * mirror_x;
            let face_y = (wrist[1] - face_center.1) * aspect / face_unit;
            hands.push(HandPose {
                handedness,
                user_hand: handedness,
                wrist_x: wrist[0],
                wrist_y: wrist[1],
                wrist_depth,
                face_x,
                face_y,
                direction: to_body(segment(0, 9)),
                across: to_body(segment(17, 5)),
                fingers: [0.0; ARTICULATION_SIZE],
                gesture: HandGesture::None,
                gesture_confidence: 0.0,
                landmarks: Some(*world),
            });
        }
        self.assign_sides(&mut hands, time);
        for hand in &mut hands {
            let side = hand.handedness.index();
            self.sides[side].last_seen = time;
            self.measure_fingers(hand, time);
            self.smooth_hand(hand, time);
        }
        self.refresh_visible(time, true);
    }

    fn assign_sides(&mut self, hands: &mut [HandPose], time: f64) {
        let labels_differ = hands.len() == 2 && hands[0].handedness != hands[1].handedness;
        if hands.len() == 2 {
            self.label_trust = (self.label_trust + if labels_differ { 0.2 } else { -0.2 }).clamp(-1.0, 1.0);
        }
        let use_label = if hands.len() == 1 { self.label_trust >= 0.0 } else { labels_differ };
        let cost = |hands: &[HandPose], index: usize, side: Side| -> f32 {
            let hand = &hands[index];
            let track = &self.sides[side.index()].track;
            let age = time - track.time;
            let tracked = age <= TRACK_MEMORY_MS;
            let mut value;
            if tracked {
                let ahead = (age.min(160.0) / 1000.0) as f32;
                value = (hand.face_x - (track.x + track.vx * ahead)).hypot(hand.face_y - (track.y + track.vy * ahead)).min(TRACK_MAX_DISTANCE);
                value += TRACK_STALE_MAX.min(((age - 70.0).max(0.0) / 100.0) as f32 * TRACK_STALE_COST);
            } else {
                let outward = if side == Side::Left { hand.face_x } else { -hand.face_x };
                value = FRESH_TRACK_COST + (0.6 - outward).max(0.0) * 0.8;
            }
            if use_label && hand.handedness != side {
                value += (if tracked { TRACKED_LABEL_WEIGHT } else { LABEL_WEIGHT }) * self.label_scores[index];
            }
            value
        };
        if hands.len() == 2 {
            let swap = cost(hands, 0, Side::Right) + cost(hands, 1, Side::Left) < cost(hands, 0, Side::Left) + cost(hands, 1, Side::Right);
            hands[0].handedness = if swap { Side::Right } else { Side::Left };
            hands[1].handedness = if swap { Side::Left } else { Side::Right };
        } else if hands.len() == 1 {
            hands[0].handedness = if cost(hands, 0, Side::Right) < cost(hands, 0, Side::Left) { Side::Right } else { Side::Left };
        }
        for hand in hands.iter_mut() {
            hand.user_hand = if self.settings.mirror { hand.handedness } else { hand.handedness.other() };
            let track = &mut self.sides[hand.handedness.index()].track;
            let elapsed = ((time - track.time) / 1000.0) as f32;
            if elapsed > 0.0 && elapsed < 0.3 {
                track.vx += (((hand.face_x - track.x) / elapsed).clamp(-25.0, 25.0) - track.vx) * 0.5;
                track.vy += (((hand.face_y - track.y) / elapsed).clamp(-25.0, 25.0) - track.vy) * 0.5;
            } else {
                track.vx = 0.0;
                track.vy = 0.0;
            }
            track.x = hand.face_x;
            track.y = hand.face_y;
            track.time = time;
        }
    }

    fn measure_fingers(&mut self, hand: &mut HandPose, time: f64) {
        let mirror_x = if self.settings.mirror { 1.0 } else { -1.0 };
        let smoothing = (self.settings.finger_smoothing + if self.settings.low_light { 0.15 } else { 0.0 }).clamp(0.0, 1.0) as f64;
        let state = &mut self.sides[hand.handedness.index()];
        if let Some(landmarks) = &hand.landmarks
            && hand_pose::measure_hand(landmarks, mirror_x, hand.handedness, &mut state.fingers, &mut state.features)
        {
            hand_pose::clamp_articulation(&mut state.fingers);
            state.filter.filter(&mut state.fingers, time, 6.0 - smoothing * 4.8, 0.8 - smoothing * 0.5, 1.2);
            state.classifier.update(&state.features);
        }
        hand.fingers = state.fingers;
        hand.gesture = state.classifier.gesture;
        hand.gesture_confidence = state.classifier.confidence;
    }

    fn smooth_hand(&mut self, hand: &HandPose, time: f64) {
        let (min_cutoff, beta) = self.settings.quality.wrist_tuning();
        let calm = if self.settings.low_light { 0.65 } else if self.settings.high_light { 0.85 } else { 1.0 };
        let min_cutoff = min_cutoff * calm;
        let state = &mut self.sides[hand.handedness.index()];
        let mut face = [hand.face_x, hand.face_y];
        state.face.filter(&mut face, time, min_cutoff, beta, 1.2);
        let mut wrist = [hand.wrist_x, hand.wrist_y];
        state.wrist.filter(&mut wrist, time, min_cutoff, beta * 10.0, 1.2);
        let mut depth = [hand.wrist_depth];
        state.depth.filter(&mut depth, time, min_cutoff * 0.6, beta * 0.5, 1.2);
        let mut orientation = [hand.direction[0], hand.direction[1], hand.direction[2], hand.across[0], hand.across[1], hand.across[2]];
        state.orientation.filter(&mut orientation, time, min_cutoff + 0.4 * calm, beta * 0.6, 1.2);
        state.smoothed = Some(HandPose {
            handedness: hand.handedness,
            user_hand: hand.user_hand,
            wrist_x: wrist[0],
            wrist_y: wrist[1],
            wrist_depth: depth[0],
            face_x: face[0],
            face_y: face[1],
            direction: normalize3([orientation[0], orientation[1], orientation[2]]),
            across: normalize3([orientation[3], orientation[4], orientation[5]]),
            fingers: hand.fingers,
            gesture: hand.gesture,
            gesture_confidence: hand.gesture_confidence,
            landmarks: hand.landmarks,
        });
    }

    /// Drops sides not seen for the hold time; call on every frame (`time` in ms).
    pub fn refresh_visible(&mut self, time: f64, mut changed: bool) {
        let hold = if self.settings.low_light { 600.0 } else { 400.0 };
        for side in &mut self.sides {
            if side.smoothed.is_some() && time - side.last_seen > hold {
                side.smoothed = None;
                side.reset_filters();
                changed = true;
            }
        }
        if changed {
            self.visible.clear();
            self.visible.extend(self.sides.iter().filter_map(|s| s.smoothed.clone()));
        }
    }

    /// Finger features of one avatar side (debugging, tests).
    pub fn features(&self, side: Side) -> &HandFeatures {
        &self.sides[side.index()].features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detection(x: f32, label: Side) -> HandDetection {
        let mut d = HandDetection { label, label_score: 0.9, presence: 1.0, ..Default::default() };
        for (i, p) in d.landmarks.iter_mut().enumerate() {
            *p = [x + (i % 5) as f32 * 0.01, 0.6 - (i / 5) as f32 * 0.02, 0.0];
        }
        for (i, p) in d.world.iter_mut().enumerate() {
            *p = [(i % 5) as f32 * 0.01, -((i / 5) as f32) * 0.02, 0.0];
        }
        d
    }

    #[test]
    fn crossed_hands_keep_their_sides() {
        let mut post = HandsPost::default();
        // Mirrored: the hand on the image right (user's left... avatar's left) starts on the left side.
        let mut time = 0.0;
        let (mut a, mut b) = (0.75f32, 0.25f32);
        post.process(&[detection(a, Side::Left), detection(b, Side::Right)], 640, 480, 0.1, (0.5, 0.4), time);
        let first: Vec<_> = post.visible.iter().map(|h| (h.handedness, h.wrist_x > 0.5)).collect();
        // Move both hands across the midline in small steps; each keeps its side.
        for _ in 0..12 {
            time += 33.0;
            a -= 0.05;
            b += 0.05;
            post.process(&[detection(a, Side::Left), detection(b, Side::Right)], 640, 480, 0.1, (0.5, 0.4), time);
        }
        for (side, started_right) in first {
            let hand = post.visible.iter().find(|h| h.handedness == side).unwrap();
            assert_eq!(hand.wrist_x > 0.5, !started_right, "side {side:?} should have crossed");
        }
    }

    #[test]
    fn hands_are_held_then_dropped() {
        let mut post = HandsPost::default();
        post.process(&[detection(0.7, Side::Left)], 640, 480, 0.1, (0.5, 0.4), 0.0);
        assert_eq!(post.visible.len(), 1);
        post.refresh_visible(300.0, false);
        assert_eq!(post.visible.len(), 1);
        post.refresh_visible(500.0, false);
        assert!(post.visible.is_empty());
    }
}
