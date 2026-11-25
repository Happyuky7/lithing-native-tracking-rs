//! The compact per-frame payload sent to the web view (little-endian,
//! fixed offsets, f32 values; decoded by guest-js/decode.ts).
//!
//! ```text
//!   0 u32  magic "LTF1"
//!   4 u16  version (1)
//!   6 u16  flags (FLAG_*)
//!   8 f64  frame id
//!  16 f64  capture time, ms on the tracker's clock
//!  24 f32  capture-to-result latency ms
//!  28 f32  face model ms
//!  32 f32  hand models ms
//!  36 u8   hand count (0-2)
//!  37 u8   face provider (0 cpu, 1 directml, 2 cuda)
//!  38 u8   hand provider
//!  39 u8   reserved
//!  40 f32 x 14  TrackingData, in TrackingData::NAMES order
//!  96 f32 x 57  blendshapes, lithing_tracking::channel_name order (52 MediaPipe + 5 tongue)
//! 324 hands, 132 bytes each:
//!       u8 handedness (0 left, 1 right), u8 userHand, u8 gesture (HandGesture::ALL index), u8 reserved
//!       f32 wristX wristY wristDepth faceX faceY, direction[3], across[3], fingers[20], gestureConfidence
//!     then, with FLAG_HAND_LANDMARKS, 63 f32 world landmarks per hand
//! then, with FLAG_FACE_LANDMARKS, 478 x 3 f32 face landmarks (normalized)
//! then, with FLAG_EYES, 2 x 9 f32 eye openness, the user's right eye first:
//!       blink gap iris geometry fused output gapOpen gapClosed calibrated(0/1)
//! then, with FLAG_GAZE, 2 f32: the raw gaze x, y (before baseline and mirroring)
//! ```
//! A frame without the optional parts is 588 bytes (660 with the eyes, 668
//! with the eyes and the gaze, both sent whenever a face is tracked): under
//! Tauri's 1 KiB threshold, so it reaches the page in one evaluated message,
//! no fetch. Decoders read the parts in order and ignore what follows, so an
//! older decoder skips the eyes and the gaze.

use lithing_tracking::eyes::{EyeDebug, EyesDebug};
use lithing_tracking::gaze::GazeSample;
use lithing_tracking::{BLENDSHAPE_CHANNELS, ExecutionProvider, HandGesture, HandPose, Side, TrackingData, TrackingFrame};

pub const MAGIC: [u8; 4] = *b"LTF1";
pub const VERSION: u16 = 1;
pub const FLAG_TRACKING: u16 = 1;
pub const FLAG_FACE_UPDATED: u16 = 2;
pub const FLAG_HANDS_UPDATED: u16 = 4;
pub const FLAG_BLENDSHAPES: u16 = 8;
pub const FLAG_TONGUE: u16 = 16;
pub const FLAG_HAND_LANDMARKS: u16 = 32;
pub const FLAG_FACE_LANDMARKS: u16 = 64;
pub const FLAG_EYES: u16 = 128;
/// The raw gaze for the pupil calibration; last, so older decoders read everything before it.
pub const FLAG_GAZE: u16 = 256;
/// f32 values per eye in the eyes part.
pub const EYE_VALUES: usize = 9;
pub const HEADER_SIZE: usize = 40;
pub const TRACKING_OFFSET: usize = 40;
pub const BLENDSHAPES_OFFSET: usize = 96;
pub const HANDS_OFFSET: usize = 324;
pub const HAND_SIZE: usize = 132;

/// What to include besides the fixed part.
#[derive(Clone, Copy, Debug, Default, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase", default)]
pub struct WireOptions {
    pub hand_landmarks: bool,
    pub face_landmarks: bool,
}

fn provider_code(provider: ExecutionProvider) -> u8 {
    match provider {
        ExecutionProvider::DirectML => 1,
        ExecutionProvider::Cuda => 2,
        _ => 0,
    }
}

struct Writer<'a>(&'a mut Vec<u8>);

impl Writer<'_> {
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32s(&mut self, values: &[f32]) {
        for &v in values {
            self.f32(v);
        }
    }
}

/// Encodes one frame into `out` (cleared first; reuse the buffer between frames).
pub fn encode(frame: &TrackingFrame, providers: [ExecutionProvider; 2], options: WireOptions, out: &mut Vec<u8>) {
    out.clear();
    let mut flags = 0u16;
    if frame.tracking.is_some() {
        flags |= FLAG_TRACKING;
    }
    if frame.face_updated {
        flags |= FLAG_FACE_UPDATED;
    }
    if frame.hands_updated {
        flags |= FLAG_HANDS_UPDATED;
    }
    if frame.blendshapes.is_some() {
        flags |= FLAG_BLENDSHAPES;
    }
    if frame.tongue.is_some_and(|t| t.valid) {
        flags |= FLAG_TONGUE;
    }
    let with_hand_landmarks = options.hand_landmarks && frame.hands.iter().all(|h| h.landmarks.is_some());
    if with_hand_landmarks {
        flags |= FLAG_HAND_LANDMARKS;
    }
    let face_landmarks = if options.face_landmarks { frame.face.as_ref().filter(|f| f.landmarks.len() == 478) } else { None };
    if face_landmarks.is_some() {
        flags |= FLAG_FACE_LANDMARKS;
    }
    if frame.eyes.is_some() {
        flags |= FLAG_EYES;
    }
    if frame.gaze.is_some() {
        flags |= FLAG_GAZE;
    }
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&(frame.frame_id as f64).to_le_bytes());
    out.extend_from_slice(&frame.timestamp_ms.to_le_bytes());
    let hands = &frame.hands[..frame.hands.len().min(2)];
    let mut w = Writer(out);
    w.f32(frame.timings.latency_ms);
    w.f32(frame.timings.face_ms);
    w.f32(frame.timings.hands_ms);
    w.0.extend_from_slice(&[hands.len() as u8, provider_code(providers[0]), provider_code(providers[1]), 0]);
    w.f32s(&frame.tracking.unwrap_or_default().to_array());
    w.f32s(&frame.blendshapes.unwrap_or([0.0; BLENDSHAPE_CHANNELS]));
    for hand in hands {
        w.0.extend_from_slice(&[hand.handedness.index() as u8, hand.user_hand.index() as u8, hand.gesture.code(), 0]);
        w.f32s(&[hand.wrist_x, hand.wrist_y, hand.wrist_depth, hand.face_x, hand.face_y]);
        w.f32s(&hand.direction);
        w.f32s(&hand.across);
        w.f32s(&hand.fingers);
        w.f32(hand.gesture_confidence);
    }
    if with_hand_landmarks {
        for hand in hands {
            for point in hand.landmarks.as_ref().unwrap() {
                w.f32s(point);
            }
        }
    }
    if let Some(face) = face_landmarks {
        for point in &face.landmarks {
            w.f32s(point);
        }
    }
    if let Some(eyes) = &frame.eyes {
        for eye in [&eyes.right, &eyes.left] {
            w.f32s(&[eye.blink, eye.gap, eye.iris, eye.geometry, eye.fused, eye.output, eye.gap_open, eye.gap_closed, if eye.calibrated { 1.0 } else { 0.0 }]);
        }
    }
    if let Some(gaze) = &frame.gaze {
        w.f32s(&[gaze.x, gaze.y]);
    }
}

/// The decoded payload (Rust side, for tests and native consumers).
#[derive(Clone, Debug, PartialEq)]
pub struct Decoded {
    pub flags: u16,
    pub frame_id: u64,
    pub timestamp_ms: f64,
    pub latency_ms: f32,
    pub tracking: TrackingData,
    pub blendshapes: [f32; BLENDSHAPE_CHANNELS],
    pub hands: Vec<HandPose>,
    pub face_landmarks: Option<Vec<[f32; 3]>>,
    pub eyes: Option<EyesDebug>,
    pub gaze: Option<GazeSample>,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum DecodeError {
    #[error("payload too short")]
    Short,
    #[error("not a lithing frame")]
    Magic,
}

pub fn decode(bytes: &[u8]) -> Result<Decoded, DecodeError> {
    if bytes.len() < HANDS_OFFSET {
        return Err(DecodeError::Short);
    }
    if bytes[0..4] != MAGIC {
        return Err(DecodeError::Magic);
    }
    let f = |offset: usize| f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    let flags = u16::from_le_bytes([bytes[6], bytes[7]]);
    let count = bytes[36] as usize;
    let mut size = HANDS_OFFSET + count * HAND_SIZE;
    if flags & FLAG_HAND_LANDMARKS != 0 {
        size += count * 63 * 4;
    }
    if flags & FLAG_FACE_LANDMARKS != 0 {
        size += 478 * 12;
    }
    if flags & FLAG_EYES != 0 {
        size += 2 * EYE_VALUES * 4;
    }
    if flags & FLAG_GAZE != 0 {
        size += 8;
    }
    if bytes.len() < size {
        return Err(DecodeError::Short);
    }
    let mut tracking = [0f32; 14];
    for (i, v) in tracking.iter_mut().enumerate() {
        *v = f(TRACKING_OFFSET + i * 4);
    }
    let mut blendshapes = [0f32; BLENDSHAPE_CHANNELS];
    for (i, v) in blendshapes.iter_mut().enumerate() {
        *v = f(BLENDSHAPES_OFFSET + i * 4);
    }
    let side = |b: u8| if b == 1 { Side::Right } else { Side::Left };
    let mut hands = Vec::with_capacity(count);
    for h in 0..count {
        let o = HANDS_OFFSET + h * HAND_SIZE;
        let v = |i: usize| f(o + 4 + i * 4);
        let mut fingers = [0f32; 20];
        for (i, finger) in fingers.iter_mut().enumerate() {
            *finger = v(11 + i);
        }
        hands.push(HandPose {
            handedness: side(bytes[o]),
            user_hand: side(bytes[o + 1]),
            gesture: HandGesture::ALL.get(bytes[o + 2] as usize).copied().unwrap_or_default(),
            wrist_x: v(0),
            wrist_y: v(1),
            wrist_depth: v(2),
            face_x: v(3),
            face_y: v(4),
            direction: [v(5), v(6), v(7)],
            across: [v(8), v(9), v(10)],
            fingers,
            gesture_confidence: v(31),
            landmarks: None,
        });
    }
    let mut offset = HANDS_OFFSET + count * HAND_SIZE;
    if flags & FLAG_HAND_LANDMARKS != 0 {
        for hand in hands.iter_mut() {
            let mut points = [[0f32; 3]; 21];
            for point in points.iter_mut() {
                *point = [f(offset), f(offset + 4), f(offset + 8)];
                offset += 12;
            }
            hand.landmarks = Some(points);
        }
    }
    let face_landmarks = (flags & FLAG_FACE_LANDMARKS != 0).then(|| {
        let points = (0..478)
            .map(|i| {
                let o = offset + i * 12;
                [f(o), f(o + 4), f(o + 8)]
            })
            .collect();
        offset += 478 * 12;
        points
    });
    let eyes = (flags & FLAG_EYES != 0).then(|| {
        let eye = |o: usize| {
            let v = |i: usize| f(o + i * 4);
            EyeDebug { blink: v(0), gap: v(1), iris: v(2), geometry: v(3), fused: v(4), output: v(5), gap_open: v(6), gap_closed: v(7), calibrated: v(8) != 0.0 }
        };
        let eyes = EyesDebug { right: eye(offset), left: eye(offset + EYE_VALUES * 4) };
        offset += 2 * EYE_VALUES * 4;
        eyes
    });
    let gaze = (flags & FLAG_GAZE != 0).then(|| GazeSample { x: f(offset), y: f(offset + 4) });
    Ok(Decoded {
        flags,
        frame_id: f64::from_le_bytes(bytes[8..16].try_into().unwrap()) as u64,
        timestamp_ms: f64::from_le_bytes(bytes[16..24].try_into().unwrap()),
        latency_ms: f(24),
        tracking: TrackingData::from_array(tracking),
        blendshapes,
        hands,
        face_landmarks,
        eyes,
        gaze,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn sample() -> TrackingFrame {
        let mut blendshapes = [0f32; BLENDSHAPE_CHANNELS];
        for (i, v) in blendshapes.iter_mut().enumerate() {
            *v = i as f32 / 100.0;
        }
        let hand = |side: Side, x: f32| HandPose {
            handedness: side,
            user_hand: side.other(),
            wrist_x: x,
            wrist_y: 0.7,
            wrist_depth: 0.1,
            face_x: 1.5,
            face_y: 3.0,
            direction: [0.0, 1.0, 0.0],
            across: [1.0, 0.0, 0.0],
            fingers: std::array::from_fn(|i| i as f32 * 0.05),
            gesture: HandGesture::Victory,
            gesture_confidence: 0.8,
            landmarks: Some(std::array::from_fn(|i| [i as f32, -(i as f32), 0.5])),
        };
        TrackingFrame {
            frame_id: 1234,
            timestamp_ms: 5678.5,
            face_updated: true,
            hands_updated: true,
            tracking: Some(TrackingData { head_yaw: 0.1, mouth_open: 0.5, blink_left: 1.0, surprised: 0.25, ..Default::default() }),
            blendshapes: Some(blendshapes),
            hands: vec![hand(Side::Left, 0.3), hand(Side::Right, 0.6)],
            ..Default::default()
        }
    }

    #[test]
    fn round_trip_and_size() {
        let frame = sample();
        let mut bytes = Vec::new();
        encode(&frame, [ExecutionProvider::DirectML, ExecutionProvider::Cpu], WireOptions::default(), &mut bytes);
        assert_eq!(bytes.len(), 588);
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded.frame_id, 1234);
        assert_eq!(decoded.tracking, frame.tracking.unwrap());
        assert_eq!(decoded.blendshapes, frame.blendshapes.unwrap());
        assert_eq!(decoded.hands.len(), 2);
        let mut expected = frame.hands[1].clone();
        expected.landmarks = None;
        assert_eq!(decoded.hands[1], expected);

        encode(&frame, [ExecutionProvider::Cpu; 2], WireOptions { hand_landmarks: true, face_landmarks: false }, &mut bytes);
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded.hands[0].landmarks, frame.hands[0].landmarks);
        assert_eq!(decode(&bytes[..100]), Err(DecodeError::Short));
    }

    #[test]
    fn eyes_follow_every_other_part() {
        let mut frame = sample();
        let eye = |x: f32| EyeDebug { blink: x, gap: 0.2, iris: 0.5, geometry: 0.7, fused: 0.3, output: 0.25, gap_open: 0.28, gap_closed: 0.08, calibrated: true };
        frame.eyes = Some(EyesDebug { right: eye(0.1), left: eye(0.6) });
        frame.face = Some(lithing_tracking::FaceDetection { landmarks: (0..478).map(|i| [i as f32, 0.5, 0.0]).collect(), ..Default::default() });
        let mut bytes = Vec::new();
        encode(&frame, [ExecutionProvider::Cpu; 2], WireOptions::default(), &mut bytes);
        assert_eq!(bytes.len(), 588 + 72);
        assert_eq!(decode(&bytes).unwrap().eyes, frame.eyes);
        encode(&frame, [ExecutionProvider::Cpu; 2], WireOptions { hand_landmarks: true, face_landmarks: true }, &mut bytes);
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded.eyes, frame.eyes);
        assert_eq!(decoded.face_landmarks.unwrap()[477], [477.0, 0.5, 0.0]);
    }

    #[test]
    fn gaze_comes_last_and_older_readers_skip_it() {
        let mut frame = sample();
        let eye = |x: f32| EyeDebug { blink: x, gap: 0.2, iris: 0.5, geometry: 0.7, fused: 0.3, output: 0.25, gap_open: 0.28, gap_closed: 0.08, calibrated: false };
        frame.eyes = Some(EyesDebug { right: eye(0.1), left: eye(0.6) });
        frame.gaze = Some(GazeSample { x: -0.21, y: 0.13 });
        frame.face = Some(lithing_tracking::FaceDetection { landmarks: (0..478).map(|i| [i as f32, 0.5, 0.0]).collect(), ..Default::default() });
        let mut bytes = Vec::new();
        encode(&frame, [ExecutionProvider::Cpu; 2], WireOptions::default(), &mut bytes);
        assert_eq!(bytes.len(), 588 + 72 + 8);
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded.flags & FLAG_GAZE, FLAG_GAZE);
        assert_eq!(decoded.eyes, frame.eyes);
        assert_eq!(decoded.gaze, frame.gaze);
        // The gaze is the last 8 bytes: a reader that ignores the flag reads the same eyes.
        assert_eq!(&bytes[bytes.len() - 8..bytes.len() - 4], &(-0.21f32).to_le_bytes());
        let mut old = bytes[..bytes.len() - 8].to_vec();
        old[6..8].copy_from_slice(&(decoded.flags & !FLAG_GAZE).to_le_bytes());
        let older = decode(&old).unwrap();
        assert_eq!(older.eyes, frame.eyes);
        assert_eq!(older.gaze, None);
        // With every optional part.
        encode(&frame, [ExecutionProvider::Cpu; 2], WireOptions { hand_landmarks: true, face_landmarks: true }, &mut bytes);
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded.eyes, frame.eyes);
        assert_eq!(decoded.gaze, frame.gaze);
        assert_eq!(decode(&bytes[..bytes.len() - 4]), Err(DecodeError::Short));
    }
}
