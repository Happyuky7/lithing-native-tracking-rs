//! What the tracker produces, named and shaped like the web app's types
//! (src/app/face-tracker.ts `TrackingData`, `HandPose`, the blendshapes
//! record) so the web view can consume native results unchanged.

use crate::geometry::Rect;

/// The 52 categories of MediaPipe's face blendshapes model, in output order.
pub const BLENDSHAPE_NAMES: [&str; 52] = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft",
    "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
];

/// Extra channels the tongue detector adds after the 52 model categories.
pub const TONGUE_NAMES: [&str; 5] = ["tongueOut", "tongueUp", "tongueDown", "tongueLeft", "tongueRight"];

/// Channels of [`TrackingFrame::blendshapes`]: the 52 model categories, then the 5 tongue channels.
pub const BLENDSHAPE_CHANNELS: usize = 57;

/// Name of blendshape channel `index` (0..57).
pub fn channel_name(index: usize) -> &'static str {
    if index < 52 { BLENDSHAPE_NAMES[index] } else { TONGUE_NAMES[index - 52] }
}

pub fn blendshape_index(name: &str) -> Option<usize> {
    BLENDSHAPE_NAMES.iter().position(|n| *n == name)
}

/// `TrackingData` of the web app: head pose, eyes, mouth and a few
/// expressions, calibrated and smoothed.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct TrackingData {
    pub head_yaw: f32,
    pub head_pitch: f32,
    pub head_roll: f32,
    pub head_x: f32,
    pub head_y: f32,
    pub head_z: f32,
    pub mouth_open: f32,
    pub blink_left: f32,
    pub blink_right: f32,
    pub eye_look_x: f32,
    pub eye_look_y: f32,
    pub happy: f32,
    pub angry: f32,
    pub surprised: f32,
}

impl TrackingData {
    pub const FIELDS: usize = 14;
    pub const NAMES: [&'static str; 14] = ["headYaw", "headPitch", "headRoll", "headX", "headY", "headZ", "mouthOpen", "blinkLeft", "blinkRight", "eyeLookX", "eyeLookY", "happy", "angry", "surprised"];

    pub fn to_array(&self) -> [f32; 14] {
        [self.head_yaw, self.head_pitch, self.head_roll, self.head_x, self.head_y, self.head_z, self.mouth_open, self.blink_left, self.blink_right, self.eye_look_x, self.eye_look_y, self.happy, self.angry, self.surprised]
    }

    pub fn from_array(a: [f32; 14]) -> Self {
        Self { head_yaw: a[0], head_pitch: a[1], head_roll: a[2], head_x: a[3], head_y: a[4], head_z: a[5], mouth_open: a[6], blink_left: a[7], blink_right: a[8], eye_look_x: a[9], eye_look_y: a[10], happy: a[11], angry: a[12], surprised: a[13] }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum Side {
    #[default]
    Left,
    Right,
}

impl Side {
    pub fn other(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }
    pub fn index(self) -> usize {
        self as usize
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Right => "right",
        }
    }
}

/// Hand gestures, as named in src/app/hand-pose.ts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum HandGesture {
    #[default]
    None,
    Open,
    Fist,
    Victory,
    ThumbsUp,
    ThumbsDown,
    Pointing,
    Ok,
    Rock,
    LoveYou,
    CallMe,
    FingerHeart,
}

impl HandGesture {
    pub const ALL: [HandGesture; 12] = [Self::None, Self::Open, Self::Fist, Self::Victory, Self::ThumbsUp, Self::ThumbsDown, Self::Pointing, Self::Ok, Self::Rock, Self::LoveYou, Self::CallMe, Self::FingerHeart];
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Open => "open",
            Self::Fist => "fist",
            Self::Victory => "victory",
            Self::ThumbsUp => "thumbs-up",
            Self::ThumbsDown => "thumbs-down",
            Self::Pointing => "pointing",
            Self::Ok => "ok",
            Self::Rock => "rock",
            Self::LoveYou => "love-you",
            Self::CallMe => "call-me",
            Self::FingerHeart => "finger-heart",
        }
    }
    pub fn code(self) -> u8 {
        Self::ALL.iter().position(|g| *g == self).unwrap_or(0) as u8
    }
}

pub const ARTICULATION_SIZE: usize = 20;

/// `HandPose` of the web app (see face-tracker.ts for every field's meaning).
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct HandPose {
    /// The avatar side this hand drives.
    pub handedness: Side,
    /// The user's own hand.
    pub user_hand: Side,
    pub wrist_x: f32,
    pub wrist_y: f32,
    pub wrist_depth: f32,
    pub face_x: f32,
    pub face_y: f32,
    pub direction: [f32; 3],
    pub across: [f32; 3],
    pub fingers: [f32; ARTICULATION_SIZE],
    pub gesture: HandGesture,
    pub gesture_confidence: f32,
    /// Metric world landmarks of the last detection (image landmarks when absent).
    pub landmarks: Option<[[f32; 3]; 21]>,
}

/// Raw face model output for one frame (normalized image coordinates).
#[derive(Clone, Debug)]
pub struct FaceDetection {
    /// 478 landmarks: x, y in 0..1 of the frame, z in frame widths (MediaPipe convention).
    pub landmarks: Vec<[f32; 3]>,
    /// The 52 blendshape scores in [`BLENDSHAPE_NAMES`] order, as the model named the sides.
    pub blendshapes: [f32; 52],
    pub presence: f32,
    pub roi: Rect,
    /// Whether the face detector ran for this frame (false: tracked from the previous landmarks).
    pub detected: bool,
}

impl Default for FaceDetection {
    fn default() -> Self {
        Self { landmarks: Vec::new(), blendshapes: [0.0; 52], presence: 0.0, roi: Rect::default(), detected: false }
    }
}

/// Raw hand model output for one hand (normalized image coordinates).
#[derive(Clone, Debug, Default)]
pub struct HandDetection {
    pub landmarks: [[f32; 3]; 21],
    /// Metric, hand-centred landmarks (metres).
    pub world: [[f32; 3]; 21],
    /// MediaPipe's handedness label (selfie convention) and its score.
    pub label: Side,
    pub label_score: f32,
    pub presence: f32,
    pub roi: Rect,
    pub detected: bool,
}

/// Tongue detector reading (see tongue.rs).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct TongueReading {
    pub out: f32,
    pub up: f32,
    pub down: f32,
    pub left: f32,
    pub right: f32,
    pub raw: f32,
    pub contrast: f32,
    pub valid: bool,
    pub reach: f32,
    pub band: f32,
    pub fill: f32,
    /// Gap between the inner lips (mouth widths), and the share of the opening that is dark / teeth.
    pub open: f32,
    pub dark: f32,
    pub teeth: f32,
    /// Mean skin lightness of the crop (0-1) and the colour noise on the skin.
    pub light: f32,
    pub noise: f32,
    /// 0-1: how much the colour test is trusted on this frame (contrast, light, noise, head turn, mouth size).
    pub confidence: f32,
}

/// Per-stage timings of one processed frame, in milliseconds.
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct FrameTimings {
    pub face_ms: f32,
    pub face_detector_ms: f32,
    pub hands_ms: f32,
    pub palm_detector_ms: f32,
    pub post_ms: f32,
    /// From capture to the result being ready.
    pub latency_ms: f32,
    /// Time the frame waited for a worker after capture.
    pub queue_ms: f32,
    /// Low / high light metering and image filter before the face models (0 when off).
    pub filter_ms: f32,
}

/// Everything the tracker produced for one camera frame. Face and hands in
/// one result always come from this same frame (`frame_id`), so they cannot
/// drift apart like the web worker experiment did.
#[derive(Clone, Debug, Default)]
pub struct TrackingFrame {
    pub frame_id: u64,
    /// Capture time, milliseconds since the tracker started.
    pub timestamp_ms: f64,
    pub width: u32,
    pub height: u32,
    /// Whether the face / hands were detected on this frame (else the values are held from earlier frames).
    pub face_updated: bool,
    pub hands_updated: bool,
    /// Calibrated, smoothed values (None while no face has been seen).
    pub tracking: Option<TrackingData>,
    /// Blendshapes named from the avatar's side (see [`channel_name`]); None while no face.
    /// The tongue channels are 0 unless tongue tracking is on.
    pub blendshapes: Option<[f32; BLENDSHAPE_CHANNELS]>,
    pub hands: Vec<HandPose>,
    /// Raw model outputs of this frame, for debugging and accuracy checks.
    pub face: Option<FaceDetection>,
    pub raw_hands: Vec<HandDetection>,
    pub tongue: Option<TongueReading>,
    /// Per eye: the raw cues (blendshape, lid gap, iris), the fused closure and the levels in use (None while no face).
    pub eyes: Option<crate::eyes::EyesDebug>,
    /// The raw gaze before baseline and mirroring (None while no face): what the pupil calibration captures.
    pub gaze: Option<crate::gaze::GazeSample>,
    /// The light filter the face models saw this frame through (low / high light mode, face frames only).
    pub light: Option<crate::light::LightState>,
    pub timings: FrameTimings,
}
