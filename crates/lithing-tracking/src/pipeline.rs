//! The tracker: a capture thread, a face thread and a hand thread.
//!
//! ```text
//! capture ──newest frame only──▶ face thread (coordinator) ──▶ results (bounded, newest wins)
//!                                   │ same Arc<Frame> ▲
//!                                   ▼                 │
//!                                 hand thread ────────┘
//! ```
//!
//! The capture thread publishes into a one-slot mailbox, so a slow model
//! skips stale frames instead of falling behind. The coordinator decides per
//! frame whether the face and/or the hands are due (each at its own rate),
//! hands the frame to the hand thread, runs the face itself in parallel,
//! waits for the hands and emits one [`TrackingFrame`] for that frame id:
//! face and hands always describe the same camera frame.

use crate::face_post::{FacePost, FaceSettings};
use crate::frame::{Frame, FramePool, LatestFrame};
use crate::hands_post::{HandSettings, HandsPost};
use crate::inference::{ExecutionProvider, InferenceError, InferenceOptions};
use crate::light::{self, LightFilter, LightMode};
use crate::models::{FaceModelOptions, FaceModels, HandModelOptions, HandModels};
use crate::priority::{self, ThreadPriority};
use crate::source::{FrameSource, SourceInfo};
use crate::cheek::{CheekDetector, CheekTuning};
use crate::tongue::{TongueDetector, TongueTuning};
use crate::types::{FrameTimings, HandDetection, TrackingFrame};
use crossbeam_channel::{Receiver, Sender, bounded};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How a file is played.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pacing {
    /// Like a camera: frames come at the file's rate (times `speed`); frames the tracker cannot keep up with are dropped.
    Realtime { speed: f32 },
    /// Every frame is processed, as fast as the tracker goes (accuracy tests, throughput).
    Lockstep,
}

/// Settings that can change while tracking runs.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, rename_all = "camelCase"))]
pub struct TrackerSettings {
    /// Face detections per second (web: 5-240, default 30).
    pub face_fps: f32,
    /// Hand detections per second (web: 4-60).
    pub hand_fps: f32,
    pub hands_enabled: bool,
    pub tongue_enabled: bool,
    pub tongue_tuning: TongueTuning,
    /// Tongue detections per second (web: 15).
    pub tongue_fps: f32,
    /// Cheek puff calibration (the cheek detector always runs, on the landmarks).
    pub cheek_tuning: CheekTuning,
    pub face: FaceSettings,
    pub hands: HandSettings,
    /// Include raw landmarks in every result (debug views, accuracy checks).
    pub include_raw: bool,
}

impl Default for TrackerSettings {
    fn default() -> Self {
        Self {
            face_fps: 30.0,
            hand_fps: 30.0,
            hands_enabled: true,
            tongue_enabled: false,
            tongue_tuning: TongueTuning::default(),
            tongue_fps: 15.0,
            cheek_tuning: CheekTuning::default(),
            face: FaceSettings::default(),
            hands: HandSettings::default(),
            include_raw: true,
        }
    }
}

/// Fixed at start.
#[derive(Clone, Debug)]
pub struct TrackerConfig {
    pub models_dir: PathBuf,
    pub inference: InferenceOptions,
    pub face_model: FaceModelOptions,
    pub hand_model: HandModelOptions,
    /// Run face and hands of one frame at the same time on two threads (else one after the other).
    pub parallel: bool,
    pub pacing: Pacing,
    pub settings: TrackerSettings,
    pub capture_priority: ThreadPriority,
    /// Raise the inference threads to normal priority while they are being
    /// starved (detections far below the asked rate), and lower them again
    /// once the load is gone. Without it, a game keeping every core busy at
    /// normal priority stops below-normal tracking almost completely.
    pub adaptive_priority: bool,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            models_dir: crate::models::default_model_dir(),
            inference: InferenceOptions::default(),
            face_model: FaceModelOptions::default(),
            hand_model: HandModelOptions::default(),
            parallel: true,
            pacing: Pacing::Realtime { speed: 1.0 },
            settings: TrackerSettings::default(),
            capture_priority: ThreadPriority::Normal,
            adaptive_priority: true,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TrackerError {
    #[error(transparent)]
    Inference(#[from] InferenceError),
    #[error(transparent)]
    Source(#[from] crate::source::SourceError),
    #[error("thread: {0}")]
    Thread(String),
}

/// Counters for benchmarks and the debug view.
#[derive(Debug, Default)]
pub struct TrackerStats {
    /// Whether the inference threads are currently raised to normal priority (adaptive priority).
    pub boosted: AtomicBool,
    /// How many times they were raised.
    pub boosts: AtomicU64,
    pub captured: AtomicU64,
    /// Frames replaced in the mailbox before the tracker took them.
    pub dropped: AtomicU64,
    pub processed: AtomicU64,
    pub face_runs: AtomicU64,
    pub hand_runs: AtomicU64,
    pub errors: AtomicU64,
    pub last_error: Mutex<Option<String>>,
}

/// What the tracker ended up running on.
#[derive(Clone, Debug, Default)]
pub struct TrackerInfo {
    pub source: SourceInfo,
    pub face_providers: Vec<ExecutionProvider>,
    pub hand_providers: Vec<ExecutionProvider>,
    pub load_ms: f32,
}

enum Command {
    Settings(Box<TrackerSettings>),
    Calibrate,
    Capture(CaptureKind, u32, Sender<Option<f32>>),
}

/// What [`Tracker::capture_level`] averages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub enum CaptureKind {
    /// The tongue detector's raw extension (tongue in / out calibration; tongue tracking must be on).
    Tongue,
    /// The cheek detector's raw width (cheeks relaxed / puffed calibration).
    Cheeks,
}

/// A running calibration capture: the sum of the samples so far.
struct Capture {
    kind: CaptureKind,
    remaining: u32,
    samples: u32,
    sum: f32,
    reply: Sender<Option<f32>>,
}

/// A running tracker. Dropping it stops every thread.
pub struct Tracker {
    results: Receiver<TrackingFrame>,
    latest: Arc<Mutex<Option<TrackingFrame>>>,
    commands: Sender<Command>,
    stop: Arc<AtomicBool>,
    mailbox: Arc<LatestFrame>,
    threads: Vec<JoinHandle<()>>,
    pub stats: Arc<TrackerStats>,
    pub info: TrackerInfo,
    pub epoch: Instant,
}

struct HandJob {
    frame: Arc<Frame>,
    allow_palm: bool,
}

struct HandReply {
    hands: Result<Vec<HandDetection>, InferenceError>,
    ms: f32,
    palm_ms: f32,
}

impl Tracker {
    /// Loads the models (on the calling thread, so errors come back here) and starts tracking `source`.
    pub fn start(source: Box<dyn FrameSource>, config: TrackerConfig) -> Result<Self, TrackerError> {
        let load_start = Instant::now();
        let face_models = FaceModels::load(&config.models_dir, &config.inference, config.face_model.clone())?;
        let hand_models = HandModels::load(&config.models_dir, &config.inference, config.hand_model.clone())?;
        let info = TrackerInfo {
            source: source.info(),
            face_providers: face_models.providers().to_vec(),
            hand_providers: hand_models.providers().to_vec(),
            load_ms: load_start.elapsed().as_secs_f32() * 1000.0,
        };
        let stop = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(TrackerStats::default());
        let mailbox = Arc::new(LatestFrame::default());
        let (result_tx, result_rx) = bounded::<TrackingFrame>(4);
        let (command_tx, command_rx) = bounded::<Command>(16);
        let latest = Arc::new(Mutex::new(None));
        let epoch = Instant::now();
        let mut threads = Vec::new();

        // Capture.
        {
            let (stop, stats, mailbox) = (stop.clone(), stats.clone(), mailbox.clone());
            let pacing = config.pacing;
            let priority = config.capture_priority;
            threads.push(
                std::thread::Builder::new()
                    .name("lithing-capture".into())
                    .spawn(move || {
                        priority::set_current(priority);
                        capture_loop(source, &mailbox, &stats, &stop, pacing, epoch);
                        mailbox.close();
                    })
                    .map_err(|e| TrackerError::Thread(e.to_string()))?,
            );
        }

        // Hands.
        let (hand_tx, hand_rx) = bounded::<HandJob>(1);
        let (hand_reply_tx, hand_reply_rx) = bounded::<HandReply>(1);
        {
            let priority = config.inference.priority;
            let hand_stats = stats.clone();
            let mut models = hand_models;
            threads.push(
                std::thread::Builder::new()
                    .name("lithing-hands".into())
                    .spawn(move || {
                        priority::set_current(priority);
                        let mut applied = false;
                        for job in hand_rx.iter() {
                            let boosted = hand_stats.boosted.load(Ordering::Relaxed);
                            if boosted != applied {
                                applied = boosted;
                                priority::set_current(if boosted { ThreadPriority::Normal.max(priority) } else { priority });
                            }
                            let start = Instant::now();
                            let hands = models.process(&job.frame, job.allow_palm);
                            let reply = HandReply { hands, ms: start.elapsed().as_secs_f32() * 1000.0, palm_ms: models.last_palm_ms };
                            if hand_reply_tx.send(reply).is_err() {
                                break;
                            }
                        }
                    })
                    .map_err(|e| TrackerError::Thread(e.to_string()))?,
            );
        }

        // Face + coordination.
        {
            let (stop, stats, mailbox, latest) = (stop.clone(), stats.clone(), mailbox.clone(), latest.clone());
            let result_rx_drain = result_rx.clone();
            let priority = config.inference.priority;
            let parallel = config.parallel;
            let adaptive = config.adaptive_priority;
            let source_fps = info.source.fps.max(1.0);
            let settings = config.settings.clone();
            let frame_period = 1000.0 / info.source.fps.max(1.0) as f64;
            let mut coordinator = Coordinator {
                face_models,
                face_post: FacePost::new(settings.face.clone()),
                hands_post: HandsPost::new(settings.hands.clone()),
                tongue: TongueDetector::default(),
                cheek: CheekDetector::default(),
                capture: None,
                settings,
                next_face_due: 0.0,
                next_hand_due: 0.0,
                next_tongue_due: 0.0,
                last_hands: Vec::new(),
                last_tracking: None,
                slack: (frame_period * 0.5).min(12.0),
                light: LightFilter::new(),
                face_input_size: (0, 0),
            };
            coordinator.hands_post.set_rate(coordinator.settings.hand_fps);
            coordinator.apply_face_confidence();
            threads.push(
                std::thread::Builder::new()
                    .name("lithing-face".into())
                    .spawn(move || {
                        priority::set_current(priority);
                        let mut governor = PriorityGovernor::new(priority, adaptive);
                        while !stop.load(Ordering::Relaxed) {
                            governor.tick(&stats, coordinator.expected_face_rate(source_fps));
                            for command in command_rx.try_iter() {
                                coordinator.apply(command);
                            }
                            let Some(frame) = mailbox.take(Duration::from_millis(50)) else {
                                if mailbox.is_closed() {
                                    break;
                                }
                                continue;
                            };
                            match coordinator.process(&frame, &hand_tx, &hand_reply_rx, parallel, &stats, epoch) {
                                Some(result) => {
                                    stats.processed.fetch_add(1, Ordering::Relaxed);
                                    *latest.lock().unwrap_or_else(|e| e.into_inner()) = Some(result.clone());
                                    // Newest wins: drop the oldest result when the consumer lags.
                                    if let Err(crossbeam_channel::TrySendError::Full(result)) = result_tx.try_send(result) {
                                        let _ = result_rx_drain.try_recv();
                                        let _ = result_tx.try_send(result);
                                    }
                                }
                                None => continue,
                            }
                        }
                        drop(hand_tx);
                    })
                    .map_err(|e| TrackerError::Thread(e.to_string()))?,
            );
        }

        Ok(Self { results: result_rx, latest, commands: command_tx, stop, mailbox, threads, stats, info, epoch })
    }

    /// Results in frame order (bounded; the oldest is dropped when not read in time).
    pub fn results(&self) -> &Receiver<TrackingFrame> {
        &self.results
    }

    /// The most recent result.
    pub fn latest(&self) -> Option<TrackingFrame> {
        self.latest.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn update_settings(&self, settings: TrackerSettings) {
        let _ = self.commands.send(Command::Settings(Box::new(settings)));
    }

    /// Re-learns the neutral pose over the next 24 face frames.
    pub fn calibrate(&self) {
        let _ = self.commands.send(Command::Calibrate);
    }

    /// Averages the raw level of the next `samples` tongue detections or face
    /// frames (calibration: tongue in / out, cheeks relaxed / puffed, as the
    /// web detectors' captureLevel). The receiver gets the average once they
    /// are in, or None when a newer capture replaces this one; the caller
    /// sets its own timeout (no face, tongue tracking off).
    pub fn capture_level(&self, kind: CaptureKind, samples: u32) -> Receiver<Option<f32>> {
        let (tx, rx) = bounded(1);
        let _ = self.commands.send(Command::Capture(kind, samples.max(1), tx));
        rx
    }

    /// Whether the source ended (files) and every frame has been processed.
    pub fn finished(&self) -> bool {
        self.threads.iter().all(|t| t.is_finished())
    }

    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.mailbox.close();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

impl Drop for Tracker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn capture_loop(mut source: Box<dyn FrameSource>, mailbox: &LatestFrame, stats: &TrackerStats, stop: &AtomicBool, pacing: Pacing, epoch: Instant) {
    let pool = FramePool::new(6);
    let live = source.is_live();
    let fps = source.info().fps.max(1.0) as f64;
    let mut id = 0u64;
    let start = Instant::now();
    while !stop.load(Ordering::Relaxed) {
        if !live && let Pacing::Realtime { speed } = pacing {
            // A file plays like a camera: frame `id` is due at id / fps.
            let due = start + Duration::from_secs_f64(id as f64 / fps / speed.max(0.01) as f64);
            let now = Instant::now();
            if due > now {
                std::thread::sleep(due - now);
            }
        }
        let mut frame = match source.next_frame(&pool, id) {
            Ok(Some(frame)) => frame,
            Ok(None) => break,
            // A pushed source between frames: ask again (and see `stop`).
            Err(crate::source::SourceError::NoFrame) => continue,
            Err(error) => {
                stats.errors.fetch_add(1, Ordering::Relaxed);
                *stats.last_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(error.to_string());
                if live {
                    std::thread::sleep(Duration::from_millis(20));
                    continue;
                }
                break;
            }
        };
        frame.captured = Instant::now();
        frame.pts_ms = if live { frame.captured.duration_since(epoch).as_secs_f64() * 1000.0 } else { id as f64 * 1000.0 / fps };
        id += 1;
        stats.captured.fetch_add(1, Ordering::Relaxed);
        if mailbox.publish(Arc::new(frame)) {
            stats.dropped.fetch_add(1, Ordering::Relaxed);
        }
        if pacing == Pacing::Lockstep && !live {
            while !mailbox.wait_taken(Duration::from_millis(100)) {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
            }
        }
    }
}

struct Coordinator {
    face_models: FaceModels,
    face_post: FacePost,
    hands_post: HandsPost,
    tongue: TongueDetector,
    cheek: CheekDetector,
    capture: Option<Capture>,
    settings: TrackerSettings,
    next_face_due: f64,
    next_hand_due: f64,
    next_tongue_due: f64,
    last_hands: Vec<HandDetection>,
    last_tracking: Option<crate::types::TrackingData>,
    slack: f64,
    /// The low / high light image processing of the face input (see light.rs).
    light: LightFilter,
    /// Size of the image the face models saw last (the 640x480 light canvas or the frame).
    face_input_size: (u32, u32),
}

/// One interval on from when the last run was due, never sooner than half an interval (web nextDue).
fn next_due(due: f64, now: f64, interval: f64) -> f64 {
    (due + interval).max(now + interval * 0.5)
}

impl Coordinator {
    /// Face detections per second the settings ask for on this source.
    fn expected_face_rate(&self, source_fps: f32) -> f32 {
        self.settings.face_fps.min(source_fps)
    }

    /// Adds one sample to a running capture of `kind`, and replies once it has them all.
    fn add_capture_sample(&mut self, kind: CaptureKind, value: f32) {
        let Some(capture) = self.capture.as_mut().filter(|capture| capture.kind == kind) else { return };
        capture.sum += value;
        capture.samples += 1;
        capture.remaining = capture.remaining.saturating_sub(1);
        if capture.remaining == 0
            && let Some(done) = self.capture.take()
        {
            let _ = done.reply.try_send(Some(done.sum / done.samples as f32));
        }
    }

    fn light_mode(&self) -> LightMode {
        LightMode::from_flags(self.settings.face.low_light, self.settings.face.high_light)
    }

    /// The face landmarker thresholds of the light mode (web faceConfidenceOptions).
    fn apply_face_confidence(&mut self) {
        let confidence = light::face_confidence(self.light_mode());
        let options = &mut self.face_models.options;
        options.min_detection_confidence = confidence.detection;
        options.min_presence_confidence = confidence.presence;
        options.min_tracking_confidence = confidence.tracking;
    }

    fn apply(&mut self, command: Command) {
        match command {
            Command::Settings(settings) => {
                let calibrate = settings.face.low_light != self.settings.face.low_light || settings.face.high_light != self.settings.face.high_light;
                self.face_post.settings = settings.face.clone();
                self.hands_post.settings = settings.hands.clone();
                self.hands_post.set_rate(settings.hand_fps);
                if !settings.hands_enabled {
                    self.hands_post.reset();
                    self.last_hands.clear();
                }
                if !settings.tongue_enabled {
                    self.tongue.reset();
                }
                self.settings = *settings;
                self.apply_face_confidence();
                if calibrate {
                    self.face_post.calibrate();
                }
            }
            Command::Calibrate => self.face_post.calibrate(),
            Command::Capture(kind, remaining, reply) => {
                if let Some(previous) = self.capture.take() {
                    let _ = previous.reply.try_send(None);
                }
                self.capture = Some(Capture { kind, remaining, samples: 0, sum: 0.0, reply });
            }
        }
    }

    fn process(&mut self, frame: &Arc<Frame>, hand_tx: &Sender<HandJob>, hand_rx: &Receiver<HandReply>, parallel: bool, stats: &TrackerStats, epoch: Instant) -> Option<TrackingFrame> {
        let now = frame.pts_ms;
        let face_interval = 1000.0 / self.settings.face_fps.clamp(1.0, 240.0) as f64;
        let hand_interval = 1000.0 / self.settings.hand_fps.clamp(1.0, 120.0) as f64;
        let run_face = now + self.slack >= self.next_face_due;
        let run_hands = self.settings.hands_enabled && now + self.slack >= self.next_hand_due;
        if !run_face && !run_hands {
            self.hands_post.refresh_visible(now, false);
            return None;
        }
        let queue_ms = frame.captured.elapsed().as_secs_f32() * 1000.0;
        let mut timings = FrameTimings { queue_ms, ..Default::default() };
        if run_face {
            self.next_face_due = next_due(self.next_face_due, now, face_interval);
        }
        // Hands start first on their own thread; the face runs here meanwhile.
        let mut hands_sent = false;
        if run_hands {
            self.next_hand_due = next_due(self.next_hand_due, now, hand_interval);
            if parallel && run_face {
                hands_sent = hand_tx.send(HandJob { frame: frame.clone(), allow_palm: true }).is_ok();
            }
        }
        let mut face = None;
        let mut light_state = None;
        if run_face {
            // The face models see the light-filtered 640x480 canvas in low / high light (web update()).
            let mode = self.light_mode();
            let (input, state) = self.light.process(frame, mode);
            if mode != LightMode::None {
                timings.filter_ms = state.ms;
                light_state = Some(state);
            }
            let size = (input.width, input.height);
            if size != self.face_input_size {
                // The tracked region is in the pixels of the previous input.
                self.face_models.reset();
                self.face_input_size = size;
            }
            let start = Instant::now();
            match self.face_models.process(input) {
                Ok(result) => face = result,
                Err(error) => record(stats, &error),
            }
            timings.face_ms = start.elapsed().as_secs_f32() * 1000.0;
            timings.face_detector_ms = self.face_models.last_detector_ms;
            stats.face_runs.fetch_add(1, Ordering::Relaxed);
        }
        if run_hands && !hands_sent && hand_tx.send(HandJob { frame: frame.clone(), allow_palm: true }).is_ok() {
            hands_sent = true;
        }
        let post_start = Instant::now();
        let mut tracking = None;
        if run_face {
            tracking = Some(match &face {
                Some(face) => self.face_post.update(face, frame.width, frame.height, now),
                None => {
                    let held = self.face_post.missing(now);
                    if self.face_post.blendshapes.is_none() {
                        self.tongue.reset();
                        self.cheek.reset();
                    }
                    held
                }
            });
        }
        let mut tongue = None;
        if let Some(face) = &face
            && let Some(shapes) = &mut self.face_post.blendshapes
        {
            // MediaPipe's cheekPuff hardly moves: fused with the cheek geometry (cheek.rs).
            let score = |name: &str| crate::types::blendshape_index(name).map_or(0.0, |index| shapes[index]);
            // A closing eye lifts the cheek and reads as a smile on that side (a wink, eyes shut
            // tight); that must not switch the puff off, so the smile counts less beside a closed eye.
            let beside = |smile: f32, blink: f32| smile * (1.0 - 0.85 * (blink * 2.5).clamp(0.0, 1.0));
            let smile = (beside(score("mouthSmileLeft"), score("eyeBlinkLeft")) + beside(score("mouthSmileRight"), score("eyeBlinkRight"))) / 2.0;
            let reading = self.cheek.update(&face.landmarks, frame.width as f32, frame.height as f32, score("cheekPuff"), smile, score("jawOpen"), &self.settings.cheek_tuning);
            if let Some(index) = crate::types::blendshape_index("cheekPuff") {
                shapes[index] = reading.puff;
            }
            self.add_capture_sample(CaptureKind::Cheeks, reading.raw);
        }
        if let Some(face) = &face
            && self.settings.tongue_enabled
        {
            if now >= self.next_tongue_due {
                self.next_tongue_due = next_due(self.next_tongue_due, now, 1000.0 / self.settings.tongue_fps.max(1.0) as f64);
                let reading = self.tongue.detect(frame, &face.landmarks, self.settings.face.mirror, &self.settings.tongue_tuning);
                if reading.valid {
                    self.add_capture_sample(CaptureKind::Tongue, self.tongue.last_raw);
                }
            }
            let reading = self.tongue.reading;
            if let Some(shapes) = &mut self.face_post.blendshapes {
                shapes[52] = shapes[52].max(if reading.valid { reading.out } else { 0.0 });
                shapes[53] = reading.up;
                shapes[54] = reading.down;
                shapes[55] = reading.left;
                shapes[56] = reading.right;
            }
            tongue = Some(reading);
        }
        if tracking.is_some() {
            self.last_tracking = tracking;
        }
        let mut post_ms = post_start.elapsed().as_secs_f32() * 1000.0;
        if hands_sent {
            match hand_rx.recv() {
                Ok(reply) => {
                    timings.hands_ms = reply.ms;
                    timings.palm_detector_ms = reply.palm_ms;
                    stats.hand_runs.fetch_add(1, Ordering::Relaxed);
                    match reply.hands {
                        Ok(hands) => {
                            let start = Instant::now();
                            self.hands_post.process(&hands, frame.width, frame.height, self.face_post.face_scale, self.face_post.face_center, now);
                            self.last_hands = hands;
                            post_ms += start.elapsed().as_secs_f32() * 1000.0;
                        }
                        Err(error) => record(stats, &error),
                    }
                }
                Err(_) => return None,
            }
        }
        self.hands_post.refresh_visible(now, false);
        timings.post_ms = post_ms;
        timings.latency_ms = frame.captured.elapsed().as_secs_f32() * 1000.0;
        let include_raw = self.settings.include_raw;
        Some(TrackingFrame {
            frame_id: frame.id,
            timestamp_ms: frame.captured.duration_since(epoch).as_secs_f64() * 1000.0,
            width: frame.width,
            height: frame.height,
            face_updated: run_face,
            hands_updated: hands_sent,
            tracking: self.last_tracking,
            blendshapes: self.face_post.blendshapes,
            hands: self.hands_post.visible.clone(),
            face: if include_raw { face } else { None },
            raw_hands: if include_raw && hands_sent { self.last_hands.clone() } else { Vec::new() },
            tongue,
            eyes: self.face_post.eyes_debug,
            gaze: self.face_post.gaze_debug,
            light: light_state,
            timings,
        })
    }
}

fn record(stats: &TrackerStats, error: &InferenceError) {
    stats.errors.fetch_add(1, Ordering::Relaxed);
    *stats.last_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(error.to_string());
}

/// Adaptive priority: below normal while tracking keeps up, normal while it
/// is starved. Checked once a second from the face thread.
struct PriorityGovernor {
    base: ThreadPriority,
    enabled: bool,
    boosted: bool,
    window_start: Instant,
    runs_at_start: u64,
    healthy_seconds: u32,
    starved_seconds: u32,
}

impl PriorityGovernor {
    fn new(base: ThreadPriority, enabled: bool) -> Self {
        Self { base, enabled: enabled && base < ThreadPriority::Normal, boosted: false, window_start: Instant::now(), runs_at_start: 0, healthy_seconds: 0, starved_seconds: 0 }
    }

    fn tick(&mut self, stats: &TrackerStats, expected: f32) {
        if !self.enabled {
            return;
        }
        let elapsed = self.window_start.elapsed().as_secs_f32();
        if elapsed < 1.0 {
            return;
        }
        let runs = stats.face_runs.load(Ordering::Relaxed);
        let rate = (runs - self.runs_at_start) as f32 / elapsed;
        self.window_start = Instant::now();
        self.runs_at_start = runs;
        // Starved: well under the asked rate. A file at its end or a paused camera
        // (no frames) is not starvation: only count windows where frames came.
        let captured = stats.captured.load(Ordering::Relaxed);
        let starved = captured > 0 && rate < expected * 0.6;
        if starved {
            self.starved_seconds += 1;
            self.healthy_seconds = 0;
        } else {
            self.healthy_seconds += 1;
            self.starved_seconds = 0;
        }
        if !self.boosted && self.starved_seconds >= 1 {
            self.set(stats, true);
        } else if self.boosted && self.healthy_seconds >= 20 {
            // Try yielding again; if the load is still there the next starved second raises it back.
            self.set(stats, false);
        }
    }

    fn set(&mut self, stats: &TrackerStats, boosted: bool) {
        self.boosted = boosted;
        stats.boosted.store(boosted, Ordering::Relaxed);
        if boosted {
            stats.boosts.fetch_add(1, Ordering::Relaxed);
        }
        priority::set_current(if boosted { ThreadPriority::Normal } else { self.base });
    }
}
