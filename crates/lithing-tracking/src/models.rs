//! The MediaPipe face and hand models and their tracking loops: detect
//! once, then follow the region of interest from each frame's landmarks and
//! re-detect only when tracking is lost (MediaPipe's VIDEO running mode).

use crate::frame::Frame;
use crate::geometry::{self, Border, Detection, Rect};
use crate::inference::{AdaptiveModel, ExecutionProvider, InferenceOptions, Result};
use crate::types::{FaceDetection, HandDetection, Side};
use std::f32::consts::FRAC_PI_2;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const FACE_DETECTOR: &str = "face_detector.onnx";
pub const FACE_LANDMARKS: &str = "face_landmarks_detector.onnx";
pub const FACE_BLENDSHAPES: &str = "face_blendshapes.onnx";
pub const HAND_DETECTOR: &str = "hand_detector.onnx";
pub const HAND_LANDMARKS: &str = "hand_landmarks_detector.onnx";
pub const ALL_MODELS: [&str; 5] = [FACE_DETECTOR, FACE_LANDMARKS, FACE_BLENDSHAPES, HAND_DETECTOR, HAND_LANDMARKS];

/// Where to find the models: `LITHING_MODELS`, else `models/` next to the executable or in the working directory, upwards.
pub fn default_model_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("LITHING_MODELS") {
        return PathBuf::from(dir);
    }
    let mut candidates = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        candidates.extend(exe.ancestors().skip(1).map(|p| p.join("models")));
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates.extend(cwd.ancestors().map(|p| p.join("models")));
    }
    candidates.into_iter().find(|dir| dir.join(FACE_LANDMARKS).is_file()).unwrap_or_else(|| PathBuf::from("models"))
}

/// The 146 face landmarks the blendshapes model reads (MediaPipe's kLandmarksSubsetIdxs).
pub const BLENDSHAPE_LANDMARKS: [usize; 146] = [
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146,
    148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293,
    295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398,
    400, 402, 405, 409, 415, 454, 466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
];

const FACE_DETECTOR_SIZE: usize = 128;
const FACE_LANDMARKS_SIZE: usize = 256;
const PALM_SIZE: usize = 192;
const HAND_LANDMARKS_SIZE: usize = 224;

/// Thresholds of the face model. The tracker sets the three confidences
/// from the light mode (see `light::face_confidence`); these defaults are
/// the web app's normal-light values.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, rename_all = "camelCase"))]
pub struct FaceModelOptions {
    pub min_detection_confidence: f32,
    pub min_presence_confidence: f32,
    /// MediaPipe's minTrackingConfidence (no effect with one face: see `light::FaceConfidence::tracking`).
    pub min_tracking_confidence: f32,
    /// When tracking loses the face, run the detector on the same frame
    /// instead of waiting for the next one (MediaPipe waits).
    pub redetect_same_frame: bool,
    pub blendshapes: bool,
}

impl Default for FaceModelOptions {
    fn default() -> Self {
        Self { min_detection_confidence: 0.45, min_presence_confidence: 0.5, min_tracking_confidence: 0.58, redetect_same_frame: true, blendshapes: true }
    }
}

/// Face detector + 478 landmarks + 52 blendshapes, tracking one face.
pub struct FaceModels {
    detector: AdaptiveModel,
    landmarks: AdaptiveModel,
    blendshapes: AdaptiveModel,
    anchors: Vec<(f32, f32)>,
    detector_input: Vec<f32>,
    landmarks_input: Vec<f32>,
    blendshapes_input: Vec<f32>,
    outputs: [Vec<f32>; 3],
    detections: Vec<Detection>,
    roi: Option<Rect>,
    pub options: FaceModelOptions,
    /// Duration of the last detector run (ms), 0 when it did not run.
    pub last_detector_ms: f32,
}

impl FaceModels {
    pub fn load(dir: &Path, inference: &InferenceOptions, options: FaceModelOptions) -> Result<Self> {
        let mut models = Self {
            detector: AdaptiveModel::load(&dir.join(FACE_DETECTOR), inference)?,
            landmarks: AdaptiveModel::load(&dir.join(FACE_LANDMARKS), inference)?,
            // The blendshape MLP is ~0.1 ms on a CPU; a GPU round trip costs more than it saves.
            blendshapes: AdaptiveModel::load(&dir.join(FACE_BLENDSHAPES), &InferenceOptions { provider: ExecutionProvider::Cpu, cpu_threads: 1, ..inference.clone() })?,
            anchors: geometry::ssd_anchors(FACE_DETECTOR_SIZE, &[8, 16, 16, 16]),
            detector_input: vec![0.0; FACE_DETECTOR_SIZE * FACE_DETECTOR_SIZE * 3],
            landmarks_input: vec![0.0; FACE_LANDMARKS_SIZE * FACE_LANDMARKS_SIZE * 3],
            blendshapes_input: vec![0.0; 146 * 2],
            outputs: [Vec::with_capacity(1434), Vec::with_capacity(896 * 16), Vec::with_capacity(64)],
            detections: Vec::with_capacity(32),
            roi: None,
            options,
            last_detector_ms: 0.0,
        };
        models.warm_up()?;
        Ok(models)
    }

    fn warm_up(&mut self) -> Result<()> {
        let [a, b, c] = &mut self.outputs;
        self.detector.calibrate(&self.detector_input, &[1, FACE_DETECTOR_SIZE, FACE_DETECTOR_SIZE, 3], &mut [a, b])?;
        self.landmarks.calibrate(&self.landmarks_input, &[1, FACE_LANDMARKS_SIZE, FACE_LANDMARKS_SIZE, 3], &mut [a, b, c])?;
        self.blendshapes.calibrate(&self.blendshapes_input, &[1, 146, 2], &mut [a])?;
        Ok(())
    }

    pub fn providers(&self) -> [ExecutionProvider; 2] {
        [self.detector.provider(), self.landmarks.provider()]
    }

    pub fn provider_switches(&self) -> u32 {
        self.detector.switches + self.landmarks.switches
    }

    /// Forgets the tracked face; the next frame runs the detector.
    pub fn reset(&mut self) {
        self.roi = None;
    }

    fn detect(&mut self, frame: &Frame) -> Result<Option<Rect>> {
        let start = Instant::now();
        let letterbox = Rect::letterbox(frame.width, frame.height);
        geometry::crop_to_tensor(frame, &letterbox, FACE_DETECTOR_SIZE, -1.0, 1.0, Border::Zero, &mut self.detector_input);
        let [boxes, scores, _] = &mut self.outputs;
        self.detector.run(&self.detector_input, &[1, FACE_DETECTOR_SIZE, FACE_DETECTOR_SIZE, 3], &mut [boxes, scores])?;
        let threshold = self.options.min_detection_confidence;
        geometry::decode_detections(boxes, scores, &self.anchors, 16, 6, FACE_DETECTOR_SIZE as f32, threshold, &letterbox, &mut self.detections);
        let best = geometry::weighted_nms(std::mem::take(&mut self.detections), 0.3, 1);
        self.last_detector_ms += start.elapsed().as_secs_f32() * 1000.0;
        // Eyes level: keypoint 0 to keypoint 1 along +x; 1.5x the box, square.
        Ok(best.first().map(|d| geometry::detection_rect(d, 0, 1, 0.0).transformed(1.5, 0.0, 0.0, true)))
    }

    /// Landmarks and blendshapes of the face in `frame`, or None when there is none.
    pub fn process(&mut self, frame: &Frame) -> Result<Option<FaceDetection>> {
        self.last_detector_ms = 0.0;
        let mut detected = false;
        let mut roi = match self.roi {
            Some(roi) => roi,
            None => {
                detected = true;
                match self.detect(frame)? {
                    Some(roi) => roi,
                    None => return Ok(None),
                }
            }
        };
        loop {
            if let Some(face) = self.landmarks_in(frame, roi, detected)? {
                return Ok(Some(face));
            }
            self.roi = None;
            if detected || !self.options.redetect_same_frame {
                return Ok(None);
            }
            detected = true;
            roi = match self.detect(frame)? {
                Some(roi) => roi,
                None => return Ok(None),
            };
        }
    }

    fn landmarks_in(&mut self, frame: &Frame, roi: Rect, detected: bool) -> Result<Option<FaceDetection>> {
        geometry::crop_to_tensor(frame, &roi, FACE_LANDMARKS_SIZE, 0.0, 1.0, Border::Replicate, &mut self.landmarks_input);
        let [points, flag, extra] = &mut self.outputs;
        self.landmarks.run(&self.landmarks_input, &[1, FACE_LANDMARKS_SIZE, FACE_LANDMARKS_SIZE, 3], &mut [points, flag, extra])?;
        let presence = geometry::sigmoid(flag[0]);
        if presence < self.options.min_presence_confidence {
            return Ok(None);
        }
        let (width, height) = (frame.width as f32, frame.height as f32);
        let size = FACE_LANDMARKS_SIZE as f32;
        let mut landmarks = Vec::with_capacity(478);
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for point in points.as_chunks::<3>().0 {
            let (x, y) = roi.to_image(point[0] / size, point[1] / size);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            landmarks.push([x / width, y / height, point[2] / size * roi.w / width]);
        }
        // Next frame's region: the landmarks' bounds, levelled on the eye corners 33 -> 263.
        let p = |i: usize| (landmarks[i][0] * width, landmarks[i][1] * height);
        let angle = geometry::rotation_between(p(33), p(263), 0.0);
        self.roi = Some(Rect { cx: (x0 + x1) / 2.0, cy: (y0 + y1) / 2.0, w: x1 - x0, h: y1 - y0, angle }.transformed(1.5, 0.0, 0.0, true));

        let mut blendshapes = [0.0f32; 52];
        if self.options.blendshapes {
            for (slot, &index) in self.blendshapes_input.as_chunks_mut::<2>().0.iter_mut().zip(BLENDSHAPE_LANDMARKS.iter()) {
                slot[0] = landmarks[index][0] * width;
                slot[1] = landmarks[index][1] * height;
            }
            let [scores, _, _] = &mut self.outputs;
            self.blendshapes.run(&self.blendshapes_input, &[1, 146, 2], &mut [scores])?;
            blendshapes.copy_from_slice(&scores[..52]);
        }
        Ok(Some(FaceDetection { landmarks, blendshapes, presence, roi, detected }))
    }
}

/// Thresholds and rates of the hand models (the web app's values).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, rename_all = "camelCase"))]
pub struct HandModelOptions {
    pub max_hands: usize,
    pub min_detection_confidence: f32,
    pub min_presence_confidence: f32,
    /// A palm detection overlapping a tracked hand this much is the same hand.
    pub min_tracking_iou: f32,
    /// With fewer hands tracked than `max_hands`, look for new palms every
    /// this many hand frames (MediaPipe: every frame, 1). With no hand
    /// tracked the detector always runs.
    pub palm_interval: u32,
}

impl Default for HandModelOptions {
    fn default() -> Self {
        Self { max_hands: 2, min_detection_confidence: 0.4, min_presence_confidence: 0.4, min_tracking_iou: 0.4, palm_interval: 2 }
    }
}

/// Palm detector + 21 hand landmarks (image and world), tracking up to `max_hands`.
pub struct HandModels {
    palm: AdaptiveModel,
    landmarks: AdaptiveModel,
    anchors: Vec<(f32, f32)>,
    palm_input: Vec<f32>,
    landmarks_input: Vec<f32>,
    outputs: [Vec<f32>; 4],
    detections: Vec<Detection>,
    tracks: Vec<Rect>,
    frames_since_palm: u32,
    pub options: HandModelOptions,
    pub last_palm_ms: f32,
}

impl HandModels {
    pub fn load(dir: &Path, inference: &InferenceOptions, options: HandModelOptions) -> Result<Self> {
        let mut models = Self {
            palm: AdaptiveModel::load(&dir.join(HAND_DETECTOR), inference)?,
            landmarks: AdaptiveModel::load(&dir.join(HAND_LANDMARKS), inference)?,
            anchors: geometry::ssd_anchors(PALM_SIZE, &[8, 16, 16, 16]),
            palm_input: vec![0.0; PALM_SIZE * PALM_SIZE * 3],
            landmarks_input: vec![0.0; HAND_LANDMARKS_SIZE * HAND_LANDMARKS_SIZE * 3],
            outputs: [Vec::with_capacity(2016 * 18), Vec::with_capacity(2016), Vec::with_capacity(4), Vec::with_capacity(63)],
            detections: Vec::with_capacity(64),
            tracks: Vec::with_capacity(4),
            frames_since_palm: u32::MAX,
            options,
            last_palm_ms: 0.0,
        };
        let [a, b, c, d] = &mut models.outputs;
        models.palm.calibrate(&models.palm_input, &[1, PALM_SIZE, PALM_SIZE, 3], &mut [a, b])?;
        models.landmarks.calibrate(&models.landmarks_input, &[1, HAND_LANDMARKS_SIZE, HAND_LANDMARKS_SIZE, 3], &mut [a, b, c, d])?;
        Ok(models)
    }

    pub fn providers(&self) -> [ExecutionProvider; 2] {
        [self.palm.provider(), self.landmarks.provider()]
    }

    pub fn provider_switches(&self) -> u32 {
        self.palm.switches + self.landmarks.switches
    }

    pub fn reset(&mut self) {
        self.tracks.clear();
        self.frames_since_palm = u32::MAX;
    }

    fn detect_palms(&mut self, frame: &Frame) -> Result<Vec<Rect>> {
        let start = Instant::now();
        let letterbox = Rect::letterbox(frame.width, frame.height);
        geometry::crop_to_tensor(frame, &letterbox, PALM_SIZE, 0.0, 1.0, Border::Zero, &mut self.palm_input);
        let [boxes, scores, _, _] = &mut self.outputs;
        self.palm.run(&self.palm_input, &[1, PALM_SIZE, PALM_SIZE, 3], &mut [boxes, scores])?;
        geometry::decode_detections(boxes, scores, &self.anchors, 18, 7, PALM_SIZE as f32, self.options.min_detection_confidence, &letterbox, &mut self.detections);
        let palms = geometry::weighted_nms(std::mem::take(&mut self.detections), 0.3, self.options.max_hands);
        self.last_palm_ms = start.elapsed().as_secs_f32() * 1000.0;
        // Wrist (0) to middle finger knuckle (2) pointing up; 2.6x the palm, shifted towards the fingers.
        Ok(palms.iter().map(|d| geometry::detection_rect(d, 0, 2, FRAC_PI_2).transformed(2.6, 0.0, -0.5, true)).collect())
    }

    /// Hands in `frame`. `allow_palm` lets the palm detector run if the
    /// schedule asks for it (false keeps to the tracked hands only).
    pub fn process(&mut self, frame: &Frame, allow_palm: bool) -> Result<Vec<HandDetection>> {
        self.last_palm_ms = 0.0;
        let mut regions: Vec<(Rect, bool)> = self.tracks.iter().map(|r| (*r, false)).collect();
        let missing = regions.len() < self.options.max_hands;
        let due = regions.is_empty() || self.frames_since_palm >= self.options.palm_interval.max(1).saturating_sub(1);
        if missing && allow_palm && due {
            self.frames_since_palm = 0;
            for palm in self.detect_palms(frame)? {
                if regions.len() >= self.options.max_hands {
                    break;
                }
                // New palms are checked against the tracked hands only (NMS already separated the palms).
                if regions.iter().filter(|(_, detected)| !detected).all(|(r, _)| r.iou(&palm) < self.options.min_tracking_iou) {
                    regions.push((palm, true));
                }
            }
        } else {
            self.frames_since_palm = self.frames_since_palm.saturating_add(1);
        }
        self.tracks.clear();
        let mut hands: Vec<HandDetection> = Vec::with_capacity(regions.len());
        let mut boxes: Vec<(f32, f32, f32, f32)> = Vec::with_capacity(regions.len());
        for (roi, detected) in regions {
            if let Some((hand, next)) = self.landmarks_in(frame, roi, detected)? {
                // Two regions that converged on the same hand (MediaPipe's landmark
                // deduplication): their landmark boxes overlap. Two hands held close
                // together have overlapping regions but separate landmark boxes.
                let bounds = landmark_bounds(&hand);
                if let Some(other) = boxes.iter().position(|b| geometry::iou(*b, bounds) > 0.5) {
                    if hands[other].presence >= hand.presence {
                        continue;
                    }
                    hands.remove(other);
                    boxes.remove(other);
                    self.tracks.remove(other);
                }
                hands.push(hand);
                boxes.push(bounds);
                self.tracks.push(next);
            }
        }
        Ok(hands)
    }

    fn landmarks_in(&mut self, frame: &Frame, roi: Rect, detected: bool) -> Result<Option<(HandDetection, Rect)>> {
        geometry::crop_to_tensor(frame, &roi, HAND_LANDMARKS_SIZE, 0.0, 1.0, Border::Zero, &mut self.landmarks_input);
        let [screen, presence, handedness, world] = &mut self.outputs;
        self.landmarks.run(&self.landmarks_input, &[1, HAND_LANDMARKS_SIZE, HAND_LANDMARKS_SIZE, 3], &mut [screen, presence, handedness, world])?;
        let presence = presence[0];
        if presence < self.options.min_presence_confidence {
            return Ok(None);
        }
        let (width, height) = (frame.width as f32, frame.height as f32);
        let size = HAND_LANDMARKS_SIZE as f32;
        let mut hand = HandDetection { presence, roi, detected, ..Default::default() };
        let mut pixels = [(0.0f32, 0.0f32); 21];
        let (sin, cos) = roi.angle.sin_cos();
        for index in 0..21 {
            let s = &screen[index * 3..index * 3 + 3];
            let (x, y) = roi.to_image(s[0] / size, s[1] / size);
            pixels[index] = (x, y);
            hand.landmarks[index] = [x / width, y / height, s[2] / size * roi.w / width];
            // World landmarks turn with the region (MediaPipe's WorldLandmarkProjection).
            let w = &world[index * 3..index * 3 + 3];
            hand.world[index] = [w[0] * cos - w[1] * sin, w[0] * sin + w[1] * cos, w[2]];
        }
        // The model's score is the probability of a right hand (selfie convention).
        let right = handedness[0];
        (hand.label, hand.label_score) = if right >= 0.5 { (Side::Right, right) } else { (Side::Left, 1.0 - right) };
        // Next region: from the wrist towards the middle of the knuckles, around the palm and finger bases.
        let (x0, y0) = pixels[0];
        let x1 = ((pixels[5].0 + pixels[13].0) / 2.0 + pixels[9].0) / 2.0;
        let y1 = ((pixels[5].1 + pixels[13].1) / 2.0 + pixels[9].1) / 2.0;
        let angle = geometry::rotation_between((x0, y0), (x1, y1), FRAC_PI_2);
        const PARTIAL: [usize; 12] = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18];
        let next = geometry::rotated_bounds(PARTIAL.iter().map(|&i| pixels[i]), angle).transformed(2.0, 0.0, -0.1, true);
        Ok(Some((hand, next)))
    }
}

/// Bounds (x0, y0, x1, y1) of a hand's image landmarks, normalized.
fn landmark_bounds(hand: &HandDetection) -> (f32, f32, f32, f32) {
    hand.landmarks.iter().fold((f32::MAX, f32::MAX, f32::MIN, f32::MIN), |b, p| (b.0.min(p[0]), b.1.min(p[1]), b.2.max(p[0]), b.3.max(p[1])))
}
