//! Native face and hand tracking for Vtuber-Lithing.
//!
//! The same MediaPipe models the web app runs in WebAssembly (face
//! detector, 478-point face landmarks, 52 blendshapes, palm detector, 21-point
//! hand landmarks), converted to ONNX and run with ONNX Runtime on their own
//! threads: DirectML (any DX12 GPU) or CUDA, the CPU, or Auto. The
//! post-processing is a port of the web tracker (src/app/face-tracker.ts,
//! hand-pose.ts, tongue-detector.ts) so the outputs keep the web app's names
//! and meaning: [`TrackingData`], [`HandPose`] and the blendshapes record.
//!
//! ```no_run
//! use lithing_tracking::{Tracker, TrackerConfig, source::Y4mSource};
//! let source = Y4mSource::open("clip.y4m".as_ref()).unwrap();
//! let tracker = Tracker::start(Box::new(source), TrackerConfig::default()).unwrap();
//! for frame in tracker.results().iter().take(100) {
//!     println!("{} {:?} hands {}", frame.frame_id, frame.tracking, frame.hands.len());
//! }
//! ```

// Index loops mirror the TypeScript originals line by line; keep them comparable.
#![allow(clippy::needless_range_loop)]

pub mod cheek;
pub mod eyes;
pub mod face_post;
pub mod frame;
pub mod gaze;
pub mod geometry;
pub mod hand_pose;
pub mod hands_post;
pub mod inference;
pub mod light;
pub mod models;
pub mod pipeline;
pub mod priority;
pub mod push;
pub mod source;
pub mod stress;
pub mod tongue;
pub mod types;

pub use inference::{ExecutionProvider, InferenceOptions};
pub use lithing_camera_controls as camera_controls;
pub use pipeline::{CaptureKind, Pacing, Tracker, TrackerConfig, TrackerError, TrackerSettings};
pub use priority::ThreadPriority;
pub use types::*;
