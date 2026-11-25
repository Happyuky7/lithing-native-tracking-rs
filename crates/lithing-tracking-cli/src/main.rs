//! `lithing-track`: runs lithing-tracking on a video file (or a camera) and
//! measures it. See README.md for examples.

mod bench;
mod metrics;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use lithing_tracking::inference::{ExecutionProvider, InferenceOptions};
use lithing_tracking::models::{self, HandModelOptions};
use lithing_tracking::source::{FrameSource, Y4mSource};
use lithing_tracking::{Pacing, ThreadPriority, Tracker, TrackerConfig, TrackingFrame};
use metrics::{Collector, Summary};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(name = "lithing-track", version, about = "Native Vtuber-Lithing tracking: run, dump and benchmark")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Track a video file (or `camera:N`) and print per-second stats and a summary.
    Run(RunArgs),
    /// Track every frame of a file (lockstep) and write raw landmarks/blendshapes as JSON lines.
    Dump(DumpArgs),
    /// Run the benchmark matrix (idle / CPU load / GPU load / low-end) and write a report.
    Bench(bench::BenchArgs),
    /// Apply the low / high light image filter to every frame of a file and write the result
    /// (raw RGB24) and the filter used per frame; prints the cost per frame.
    Filter(FilterArgs),
    /// List cameras (does not open them). `--controls` also reads each
    /// physical camera's exposure / gain / frame-rate controls without
    /// starting a stream (skipped while another app uses a camera, unless `--force`).
    Cameras {
        /// Also read each physical camera's controls (opens the device, never streams).
        #[arg(long)]
        controls: bool,
        /// Read the controls even while another app uses a camera.
        #[arg(long, requires = "controls")]
        force: bool,
        /// Print the controls as JSON.
        #[arg(long, requires = "controls")]
        json: bool,
        /// Also list each camera's formats (format, resolution, fps); opens the device, never streams.
        #[arg(long)]
        formats: bool,
    },
    /// Print the model files and their SHA-256.
    Models {
        #[arg(long)]
        dir: Option<PathBuf>,
    },
    /// Time each model alone on each provider (per-inference cost, no pipeline).
    ModelTimes {
        #[arg(long, default_value = "cpu,directml")]
        eps: String,
        #[arg(long, default_value_t = 1)]
        threads: usize,
        #[arg(long, default_value_t = 200)]
        runs: usize,
        #[arg(long)]
        models: Option<PathBuf>,
    },
    /// Keep N threads busy (CPU contention for benchmarks).
    BurnCpu {
        #[arg(long, default_value_t = 0)]
        threads: usize,
        #[arg(long, default_value_t = 60.0)]
        seconds: f32,
        /// Restrict to this many CPUs (see `run --affinity`).
        #[arg(long)]
        affinity: Option<usize>,
    },
    /// Keep the GPU busy with large matrix products (GPU contention for benchmarks).
    BurnGpu {
        #[arg(long, default_value_t = 2048)]
        size: usize,
        #[arg(long, default_value_t = 8)]
        depth: usize,
        #[arg(long, default_value_t = 60.0)]
        seconds: f32,
        #[arg(long, default_value = "directml")]
        ep: String,
    },
}

#[derive(Args, Clone)]
pub struct TrackArgs {
    /// Video file (.y4m directly; .mp4/.webm/... through ffmpeg) or `camera:N`.
    pub input: String,
    /// auto | directml | cuda | cpu
    #[arg(long, default_value = "auto")]
    pub ep: String,
    /// Threads per CPU session.
    #[arg(long, default_value_t = 1)]
    pub threads: usize,
    /// Inference thread priority: lowest | below-normal | normal | above-normal.
    #[arg(long, default_value = "below-normal")]
    pub priority: String,
    /// Keep --priority even when tracking is starved (no automatic raise to normal).
    #[arg(long)]
    pub fixed_priority: bool,
    #[arg(long, default_value_t = 30.0)]
    pub face_fps: f32,
    #[arg(long, default_value_t = 30.0)]
    pub hand_fps: f32,
    #[arg(long)]
    pub no_hands: bool,
    #[arg(long)]
    pub tongue: bool,
    /// Run face and hands one after the other instead of on two threads.
    #[arg(long)]
    pub serial: bool,
    /// Look for a second hand every N hand frames (1 = MediaPipe, every frame).
    #[arg(long, default_value_t = 2)]
    pub palm_interval: u32,
    /// Do not re-run the face detector on the frame where tracking was lost (MediaPipe behaviour).
    #[arg(long)]
    pub no_same_frame_redetect: bool,
    /// Restrict the process to this many CPUs (every other logical CPU, so no SMT siblings): low-end simulation.
    #[arg(long)]
    pub affinity: Option<usize>,
    #[arg(long)]
    pub models: Option<PathBuf>,
    /// Read the input as raw RGB24 frames of this size (WxH), e.g. from `ffmpeg -f rawvideo -pix_fmt rgb24`.
    #[arg(long)]
    pub raw_size: Option<String>,
    /// Frame rate of a raw RGB24 input.
    #[arg(long, default_value_t = 30.0)]
    pub raw_fps: f32,
    /// Camera: keep the frame rate in low light (the default): auto-exposure
    /// priority off, else an exposure cap with more gain; restored at the end.
    #[arg(long, overrides_with = "no_keep_fps")]
    pub keep_fps: bool,
    /// Camera: leave its exposure controls alone (it may drop to 15 fps in the dark).
    #[arg(long, overrides_with = "keep_fps")]
    pub no_keep_fps: bool,
    /// Camera mode, e.g. `MJPEG:1280x720@30` or `640x480@60` (see `cameras --formats`);
    /// default Auto. A mode the camera rejects opens in Auto with a warning.
    #[arg(long = "mode")]
    pub camera_mode: Option<String>,
    #[command(flatten)]
    pub modes: ModeArgs,
}

/// The web tracker's tracking options (Tracking panel), applied from the start.
#[derive(Args, Clone, Debug)]
pub struct ModeArgs {
    /// Low light mode (web "Mejorar camara con poca luz"): adaptive brighten + blur filter for the face, lower thresholds, calmer smoothing, longer holds.
    #[arg(long)]
    pub low_light: bool,
    /// Overexposed camera mode (web "Mejorar camara sobreexpuesta"): darkening filter for the face and its thresholds / smoothing.
    #[arg(long)]
    pub high_light: bool,
    /// Glasses mode (web "Tracking optimizado para gafas"): blinks and gaze weighted for glasses.
    #[arg(long)]
    pub glasses: bool,
    /// Do not mirror the face (web: mirrorFaceTracking off).
    #[arg(long)]
    pub no_mirror_face: bool,
    /// Do not mirror the hands (web: mirrorHandTracking off).
    #[arg(long)]
    pub no_mirror_hands: bool,
    /// Tracking stability 0..1 (web default 0.72).
    #[arg(long, default_value_t = 0.72)]
    pub stability: f32,
    /// Finger smoothing 0..1 (web default 0.5).
    #[arg(long, default_value_t = 0.5)]
    pub finger_smoothing: f32,
    /// Hand quality: performance | balanced | precise.
    #[arg(long, default_value = "balanced")]
    pub hand_quality: String,
    /// Eye openness curve 0..1 (web default 0.5).
    #[arg(long, default_value_t = 0.5)]
    pub eye_openness: f32,
    /// Eye calibration as the app stores it (JSON: right/left, each blink/gap/iris open/closed).
    #[arg(long)]
    pub eye_calibration: Option<PathBuf>,
}

impl ModeArgs {
    pub fn apply(&self, settings: &mut lithing_tracking::TrackerSettings) -> Result<()> {
        use lithing_tracking::hands_post::HandQuality;
        settings.face.low_light = self.low_light;
        settings.face.high_light = self.high_light;
        settings.hands.low_light = self.low_light;
        settings.hands.high_light = self.high_light;
        settings.face.glasses_mode = self.glasses;
        settings.face.mirror = !self.no_mirror_face;
        settings.hands.mirror = !self.no_mirror_hands;
        settings.face.stability = self.stability.clamp(0.0, 1.0);
        settings.hands.finger_smoothing = self.finger_smoothing.clamp(0.0, 1.0);
        settings.face.eye_openness = self.eye_openness.clamp(0.0, 1.0);
        if let Some(path) = &self.eye_calibration {
            let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
            settings.face.eye_calibration = Some(serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?);
        }
        settings.hands.quality = match self.hand_quality.as_str() {
            "performance" => HandQuality::Performance,
            "balanced" => HandQuality::Balanced,
            "precise" => HandQuality::Precise,
            other => bail!("unknown --hand-quality {other}"),
        };
        Ok(())
    }

    /// Short description for summaries.
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if self.low_light {
            parts.push("low light");
        }
        if self.high_light {
            parts.push("high light");
        }
        if self.glasses {
            parts.push("glasses");
        }
        if parts.is_empty() { String::new() } else { format!(", {}", parts.join(" + ")) }
    }
}

#[derive(Args)]
struct FilterArgs {
    /// Video file (or raw RGB24 with --raw-size).
    input: String,
    /// low | high
    #[arg(long, default_value = "low")]
    mode: String,
    #[arg(long)]
    raw_size: Option<String>,
    #[arg(long, default_value_t = 30.0)]
    raw_fps: f32,
    /// Filtered frames, raw RGB24 640x480 each.
    #[arg(long)]
    out: Option<PathBuf>,
    /// One JSON line per frame: luminance, filter string, ms.
    #[arg(long)]
    log: Option<PathBuf>,
    #[arg(long)]
    frames: Option<u64>,
}

#[derive(Args)]
struct RunArgs {
    #[command(flatten)]
    track: TrackArgs,
    /// Stop after this many seconds.
    #[arg(long)]
    duration: Option<f32>,
    /// Restart the file when it ends.
    #[arg(long = "loop")]
    looping: bool,
    /// Process every frame as fast as possible instead of at the file's rate.
    #[arg(long)]
    lockstep: bool,
    /// Playback speed for files (realtime pacing).
    #[arg(long, default_value_t = 1.0)]
    speed: f32,
    /// Write every result as a JSON line.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Write the summary as JSON.
    #[arg(long)]
    summary_json: Option<PathBuf>,
    /// Label stored in the summary.
    #[arg(long, default_value = "run")]
    label: String,
    #[arg(long)]
    quiet: bool,
}

#[derive(Args)]
struct DumpArgs {
    #[command(flatten)]
    track: TrackArgs,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    frames: Option<u64>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run(args),
        Command::Dump(args) => dump(args),
        Command::Bench(args) => bench::bench(args),
        Command::Filter(args) => filter(args),
        Command::Cameras { controls, force, json, formats } => cameras(controls, force, json, formats),
        Command::Models { dir } => list_models(dir),
        Command::ModelTimes { eps, threads, runs, models } => model_times(&eps, threads, runs, models),
        Command::BurnCpu { threads, seconds, affinity } => burn_cpu(threads, seconds, affinity),
        Command::BurnGpu { size, depth, seconds, ep } => burn_gpu(size, depth, seconds, &ep),
    }
}

/// Pins the process to `count` CPUs, skipping SMT siblings (0, 2, 4, ...).
pub fn set_affinity(count: usize) -> Result<()> {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Threading::{GetCurrentProcess, SetProcessAffinityMask};
        let mut mask = 0usize;
        for i in 0..count {
            mask |= 1 << (i * 2);
        }
        if SetProcessAffinityMask(GetCurrentProcess(), mask) == 0 {
            bail!("SetProcessAffinityMask failed");
        }
    }
    #[cfg(not(windows))]
    {
        let cpus: Vec<String> = (0..count).map(|i| (i * 2).to_string()).collect();
        eprintln!("affinity on Linux: run under `taskset -c {}`", cpus.join(","));
    }
    Ok(())
}

pub fn tracker_config(args: &TrackArgs, pacing: Pacing) -> Result<TrackerConfig> {
    let provider = ExecutionProvider::parse(&args.ep).with_context(|| format!("unknown --ep {}", args.ep))?;
    let priority = ThreadPriority::parse(&args.priority).with_context(|| format!("unknown --priority {}", args.priority))?;
    let mut config = TrackerConfig {
        models_dir: args.models.clone().unwrap_or_else(models::default_model_dir),
        inference: InferenceOptions { provider, cpu_threads: args.threads, priority, ..Default::default() },
        hand_model: HandModelOptions { palm_interval: args.palm_interval, ..Default::default() },
        parallel: !args.serial,
        adaptive_priority: !args.fixed_priority,
        pacing,
        ..Default::default()
    };
    config.face_model.redetect_same_frame = !args.no_same_frame_redetect;
    config.settings.face_fps = args.face_fps;
    config.settings.hand_fps = args.hand_fps;
    config.settings.hands_enabled = !args.no_hands;
    config.settings.tongue_enabled = args.tongue;
    args.modes.apply(&mut config.settings)?;
    Ok(config)
}

fn parse_size(size: &str) -> Result<(u32, u32)> {
    let (w, h) = size.split_once('x').context("--raw-size wants WxH")?;
    Ok((w.parse().context("width")?, h.parse().context("height")?))
}

fn open_input(args: &TrackArgs, looping: bool) -> Result<Box<dyn FrameSource>> {
    let mode = match &args.camera_mode {
        Some(text) => Some(lithing_tracking::source::CameraMode::parse(text).with_context(|| format!("--mode {text}: want FORMAT:WxH@FPS or WxH@FPS"))?),
        None => None,
    };
    if mode.is_some() && args.raw_size.is_none() {
        return open_source(&args.input, looping, !args.no_keep_fps, mode);
    }
    open_any(&args.input, args.raw_size.as_deref(), args.raw_fps, looping, !args.no_keep_fps)
}

fn open_any(input: &str, raw_size: Option<&str>, raw_fps: f32, looping: bool, keep_fps: bool) -> Result<Box<dyn FrameSource>> {
    if let Some(size) = raw_size {
        let (w, h) = parse_size(size)?;
        return Ok(Box::new(lithing_tracking::source::RawRgbSource::open(input.as_ref(), w, h, raw_fps).with_context(|| format!("opening {input}"))?));
    }
    open_source(input, looping, keep_fps, None)
}

fn open_source(input: &str, looping: bool, keep_fps: bool, mode: Option<lithing_tracking::source::CameraMode>) -> Result<Box<dyn FrameSource>> {
    if let Some(index) = input.strip_prefix("camera:") {
        #[cfg(feature = "camera")]
        {
            let request = lithing_tracking::source::CameraRequest { index: index.parse().context("camera index")?, keep_frame_rate: keep_fps, mode, ..Default::default() };
            let source = lithing_tracking::source::CameraSource::open(&request)?;
            if let Some(warning) = source.info().warning {
                eprintln!("warning: {warning}");
            }
            return Ok(Box::new(source));
        }
        #[cfg(not(feature = "camera"))]
        {
            let _ = (keep_fps, mode);
            bail!("built without the camera feature ({index})");
        }
    }
    let path = PathBuf::from(input);
    let mut source = Y4mSource::open(&path).with_context(|| format!("opening {input}"))?;
    if looping {
        source.looping = Some(path);
    }
    Ok(Box::new(source))
}

fn run(args: RunArgs) -> Result<()> {
    if let Some(count) = args.track.affinity {
        set_affinity(count)?;
    }
    let pacing = if args.lockstep { Pacing::Lockstep } else { Pacing::Realtime { speed: args.speed } };
    let config = tracker_config(&args.track, pacing)?;
    let source = open_input(&args.track, args.looping)?;
    let camera_controls = source.camera_controls();
    let tracker = Tracker::start(source, config).context("starting the tracker")?;
    let info = tracker.info.clone();
    if !args.quiet {
        eprintln!(
            "source {}x{} @ {:.1} fps {} ({}), face on {:?}, hands on {:?}, models loaded in {:.0} ms",
            info.source.width, info.source.height, info.source.fps, info.source.format, info.source.description, info.face_providers, info.hand_providers, info.load_ms
        );
    }
    let mut out = match &args.out {
        Some(path) => Some(std::io::BufWriter::new(std::fs::File::create(path)?)),
        None => None,
    };
    let mut collector = Collector::start(&args.label, &args.track, &info);
    let start = Instant::now();
    let deadline = args.duration.map(|s| start + Duration::from_secs_f32(s));
    let mut last_report = Instant::now();
    loop {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            break;
        }
        match tracker.results().recv_timeout(Duration::from_millis(200)) {
            Ok(frame) => {
                collector.add(&frame);
                if let Some(out) = &mut out {
                    writeln!(out, "{}", frame_json(&frame))?;
                    // Flushed per frame: a live reader (tools/vrchat-osc-bridge.mjs) tails this file.
                    out.flush()?;
                }
            }
            Err(crossbeam_timeout) => {
                let _ = crossbeam_timeout;
                if tracker.finished() {
                    break;
                }
            }
        }
        if !args.quiet && last_report.elapsed() >= Duration::from_secs(1) {
            last_report = Instant::now();
            match &camera_controls {
                Some(handle) => {
                    let report = handle.report();
                    let camera = report.fps_after.or(report.fps_before).map(|fps| format!("  camera {fps:4.1} fps ({})", report.strategy)).unwrap_or_default();
                    eprintln!("{}{camera}", collector.progress_line());
                }
                None => eprintln!("{}", collector.progress_line()),
            }
        }
    }
    let summary = collector.finish(&tracker);
    // Closes the camera: its controls are restored before this returns.
    drop(tracker);
    if !args.quiet {
        println!("{}", summary.table_header());
        println!("{}", summary.table_row());
        if let Some(handle) = &camera_controls {
            print_keep_fps_report(&handle.report());
        }
    }
    if let Some(path) = &args.summary_json {
        std::fs::write(path, serde_json::to_string_pretty(&summary)?)?;
    }
    Ok(())
}

fn round(v: f32, digits: i32) -> f64 {
    let scale = 10f64.powi(digits);
    (v as f64 * scale).round() / scale
}

fn frame_json(frame: &TrackingFrame) -> serde_json::Value {
    let blendshapes = frame.blendshapes.map(|values| {
        values.iter().enumerate().map(|(i, v)| (lithing_tracking::channel_name(i).to_string(), serde_json::json!(round(*v, 4)))).collect::<serde_json::Map<_, _>>()
    });
    serde_json::json!({
        "frame": frame.frame_id,
        "t": (frame.timestamp_ms * 10.0).round() / 10.0,
        "face": frame.face_updated,
        "handsUpdated": frame.hands_updated,
        "tracking": frame.tracking,
        "blendshapes": blendshapes,
        "hands": frame.hands.iter().map(|h| serde_json::json!({
            "handedness": h.handedness, "userHand": h.user_hand, "wristX": h.wrist_x, "wristY": h.wrist_y, "wristDepth": h.wrist_depth,
            "faceX": h.face_x, "faceY": h.face_y, "direction": h.direction, "across": h.across, "fingers": h.fingers,
            "gesture": h.gesture, "gestureConfidence": h.gesture_confidence
        })).collect::<Vec<_>>(),
        "tongue": frame.tongue,
        "light": frame.light.map(|l| serde_json::json!({ "luminance": l.luminance, "filter": l.filter.css() })),
        "timings": frame.timings,
    })
}

fn filter(args: FilterArgs) -> Result<()> {
    use lithing_tracking::frame::FramePool;
    use lithing_tracking::light::{LightFilter, LightMode};
    let mode = match args.mode.as_str() {
        "low" => LightMode::Low,
        "high" => LightMode::High,
        other => bail!("unknown --mode {other} (low | high)"),
    };
    let mut source = open_any(&args.input, args.raw_size.as_deref(), args.raw_fps, false, true)?;
    let info = source.info();
    let pool = FramePool::new(2);
    let mut light = LightFilter::new();
    let mut out = match &args.out {
        Some(path) => Some(std::io::BufWriter::new(std::fs::File::create(path)?)),
        None => None,
    };
    let mut log = match &args.log {
        Some(path) => Some(std::io::BufWriter::new(std::fs::File::create(path)?)),
        None => None,
    };
    let mut times = Vec::new();
    let mut id = 0u64;
    while let Some(frame) = source.next_frame(&pool, id)? {
        let start = Instant::now();
        let (filtered, _) = light.process(&frame, mode);
        times.push(start.elapsed().as_secs_f32() * 1000.0);
        if let Some(out) = &mut out {
            out.write_all(&filtered.rgb)?;
        }
        if let Some(log) = &mut log {
            let state = light.last;
            writeln!(log, "{}", serde_json::json!({ "frame": id, "luminance": state.luminance, "brightness": light.brightness(), "filter": state.filter.css(), "ms": state.ms }))?;
        }
        id += 1;
        if args.frames.is_some_and(|n| id >= n) {
            break;
        }
    }
    let stat = metrics::Stat::of(&mut times);
    println!("{} frames {}x{} -> 640x480, {mode:?} light filter: mean {:.3} ms, p50 {:.3} ms, p95 {:.3} ms per frame", id, info.width, info.height, stat.mean, stat.p50, stat.p95);
    Ok(())
}

fn dump(args: DumpArgs) -> Result<()> {
    let mut config = tracker_config(&args.track, Pacing::Lockstep)?;
    // Every frame: the reference runs the models on every frame too.
    config.settings.face_fps = 1000.0;
    config.settings.hand_fps = 1000.0;
    config.settings.include_raw = true;
    let source = open_input(&args.track, false)?;
    let tracker = Tracker::start(source, config)?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args.out)?);
    let mut count = 0u64;
    let r6 = |v: f32| round(v, 6);
    while let Ok(frame) = tracker.results().recv_timeout(Duration::from_secs(10)) {
        let face = frame.face.as_ref().map(|f| {
            serde_json::json!({
                "landmarks": f.landmarks.iter().map(|p| [r6(p[0]), r6(p[1]), r6(p[2])]).collect::<Vec<_>>(),
                "blendshapes": f.blendshapes.iter().map(|v| r6(*v)).collect::<Vec<_>>(),
                "presence": f.presence, "detected": f.detected,
            })
        });
        let hands: Vec<_> = frame
            .raw_hands
            .iter()
            .map(|h| {
                serde_json::json!({
                    "landmarks": h.landmarks.iter().map(|p| [r6(p[0]), r6(p[1]), r6(p[2])]).collect::<Vec<_>>(),
                    "world": h.world.iter().map(|p| [r6(p[0]), r6(p[1]), r6(p[2])]).collect::<Vec<_>>(),
                    "label": h.label, "labelScore": h.label_score, "presence": h.presence,
                })
            })
            .collect();
        let line = serde_json::json!({ "frame": frame.frame_id, "face": face, "hands": hands, "tracking": frame.tracking, "eyes": frame.eyes, "light": frame.light.map(|l| serde_json::json!({ "luminance": l.luminance, "filter": l.filter.css(), "ms": l.ms })), "visibleHands": frame.hands.iter().map(|h| serde_json::json!({"handedness": h.handedness, "fingers": h.fingers, "gesture": h.gesture})).collect::<Vec<_>>() });
        writeln!(out, "{line}")?;
        count += 1;
        if args.frames.is_some_and(|n| count >= n) {
            break;
        }
    }
    eprintln!("wrote {count} frames to {}", args.out.display());
    Ok(())
}

fn cameras(controls: bool, force: bool, json: bool, formats: bool) -> Result<()> {
    #[cfg(feature = "camera")]
    for camera in lithing_tracking::source::list_cameras()? {
        println!("{camera}");
    }
    #[cfg(feature = "camera")]
    if formats {
        for (camera, list) in lithing_tracking::source::list_camera_formats()? {
            println!("
{camera}");
            for format in list { println!("    {format}"); }
        }
    }
    #[cfg(not(feature = "camera"))]
    println!("built without the camera feature");
    if !controls {
        return Ok(());
    }
    use lithing_tracking::camera_controls::{ControlKind, list_device_controls};
    let devices = list_device_controls(force).map_err(anyhow::Error::msg)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&devices)?);
        return Ok(());
    }
    for entry in devices {
        let device = &entry.device;
        println!("\n[{}] {}{}\n    {}", device.index, device.name, if device.virtual_camera { " (virtual)" } else { "" }, device.id);
        if let Some(why) = &entry.skipped {
            println!("    {why}");
        }
        if entry.controls.is_empty() && entry.skipped.is_none() {
            println!("    no exposure / gain / frame-rate controls");
        }
        for control in &entry.controls {
            println!("    {}", control_line(control));
        }
        if !entry.controls.is_empty() {
            let has = |kind: ControlKind| entry.controls.iter().any(|c| c.kind == kind);
            let plan = if has(ControlKind::AutoExposurePriority) || has(ControlKind::VariableFrameRate) {
                "keep-fps: turns auto-exposure priority / variable frame rate off"
            } else if entry.controls.iter().any(|c| c.kind == ControlKind::Exposure && c.supports_manual) {
                "keep-fps: no priority control; caps the exposure (and raises gain) only if the rate drops"
            } else {
                "keep-fps: nothing it can change on this camera"
            };
            println!("    {plan}");
        }
    }
    Ok(())
}

fn control_line(control: &lithing_tracking::camera_controls::ControlInfo) -> String {
    let mode = match (control.auto, control.supports_auto) {
        (true, _) => " auto",
        (false, true) => " manual (auto available)",
        _ => "",
    };
    format!("{:<24} {:>6}{mode}  range {}..{} step {} default {}  [{}]", control.kind.key(), control.value, control.min, control.max, control.step, control.default, control.name)
}

fn print_keep_fps_report(report: &lithing_tracking::camera_controls::CameraControlReport) {
    let fps = |v: Option<f32>| v.map(|f| format!("{f:.1}")).unwrap_or_else(|| "-".into());
    println!(
        "\ncamera {}: keep frame rate {}, strategy {}, negotiated {:.0} fps, delivered before {} / after {} fps",
        report.device,
        if report.keep_frame_rate { "on" } else { "off" },
        report.strategy,
        report.target_fps,
        fps(report.fps_before),
        fps(report.fps_after)
    );
    if report.virtual_camera {
        println!("  virtual camera: no hardware controls, nothing changed");
    } else if let Some(error) = &report.controls_error {
        println!("  controls: {error}");
    }
    for control in &report.controls {
        println!("  {}", control_line(control));
    }
    for change in &report.changes {
        let from = change.from.map(|v| format!("{}{}", v.value, if v.auto { " auto" } else { "" })).unwrap_or_else(|| "?".into());
        let to = format!("{}{}", change.to.value, if change.to.auto { " auto" } else { "" });
        let result = change.error.as_deref().map(|e| format!("  FAILED: {e}")).unwrap_or_default();
        println!("  {:6.1}s {} {from} -> {to} ({}){result}", change.at, change.kind.key(), change.reason);
    }
    for note in &report.notes {
        println!("  note: {note}");
    }
    if !report.changes.is_empty() {
        if report.restored {
            println!("  restored: every changed control is back to its value before the run");
        } else {
            println!("  NOT fully restored: {}", report.restore_errors.join("; "));
        }
    }
}

fn list_models(dir: Option<PathBuf>) -> Result<()> {
    use sha2::Digest;
    let dir = dir.unwrap_or_else(models::default_model_dir);
    println!("models in {}", dir.display());
    for name in models::ALL_MODELS {
        let path = dir.join(name);
        match std::fs::read(&path) {
            Ok(bytes) => println!("{:x}  {name} ({} bytes)", sha2::Sha256::digest(&bytes), bytes.len()),
            Err(e) => println!("missing  {name} ({e})"),
        }
    }
    Ok(())
}

fn model_times(eps: &str, threads: usize, runs: usize, dir: Option<PathBuf>) -> Result<()> {
    let dir = dir.unwrap_or_else(models::default_model_dir);
    let shapes: [(&str, Vec<usize>, usize); 5] = [
        (models::FACE_DETECTOR, vec![1, 128, 128, 3], 2),
        (models::FACE_LANDMARKS, vec![1, 256, 256, 3], 3),
        (models::FACE_BLENDSHAPES, vec![1, 146, 2], 1),
        (models::HAND_DETECTOR, vec![1, 192, 192, 3], 2),
        (models::HAND_LANDMARKS, vec![1, 224, 224, 3], 4),
    ];
    println!("model | provider | threads | load ms | mean ms | p50 ms | p95 ms");
    for ep in eps.split(',') {
        let provider = ExecutionProvider::parse(ep.trim()).context("unknown provider")?;
        for (name, shape, outputs) in &shapes {
            let options = InferenceOptions { provider, cpu_threads: threads, ..Default::default() };
            let start = Instant::now();
            let mut model = match lithing_tracking::inference::Model::load(&dir.join(name), provider, &options) {
                Ok(m) => m,
                Err(e) => {
                    println!("{name} | {ep} | failed: {e}");
                    continue;
                }
            };
            let load_ms = start.elapsed().as_secs_f32() * 1000.0;
            let input: Vec<f32> = (0..shape.iter().product::<usize>()).map(|i| (i % 255) as f32 / 255.0).collect();
            let mut buffers: Vec<Vec<f32>> = (0..*outputs).map(|_| Vec::new()).collect();
            let mut times = Vec::with_capacity(runs);
            for run in 0..runs + 10 {
                let mut refs: Vec<&mut Vec<f32>> = buffers.iter_mut().collect();
                let t = Instant::now();
                model.run(&input, shape, &mut refs)?;
                if run >= 10 {
                    times.push(t.elapsed().as_secs_f32() * 1000.0);
                }
            }
            let stat = metrics::Stat::of(&mut times);
            println!("{name} | {ep} | {threads} | {load_ms:.0} | {:.3} | {:.3} | {:.3}", stat.mean, stat.p50, stat.p95);
        }
    }
    Ok(())
}

fn burn_cpu(threads: usize, seconds: f32, affinity: Option<usize>) -> Result<()> {
    if let Some(count) = affinity {
        set_affinity(count)?;
    }
    let threads = if threads == 0 { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) } else { threads };
    let end = Instant::now() + Duration::from_secs_f32(seconds);
    let handles: Vec<_> = (0..threads)
        .map(|seed| {
            std::thread::spawn(move || {
                // Integer and float work with a working set larger than L2, like a game thread.
                let mut buffer = vec![seed as u64; 1 << 18];
                let mut x = seed as u64 + 1;
                let mut f = 1.0f64;
                while Instant::now() < end {
                    for _ in 0..20_000 {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        let index = (x as usize) & (buffer.len() - 1);
                        buffer[index] = buffer[index].wrapping_add(x);
                        f = (f * 1.000001 + 0.5).sqrt();
                    }
                }
                std::hint::black_box((buffer[0], f));
            })
        })
        .collect();
    for handle in handles {
        let _ = handle.join();
    }
    Ok(())
}

fn burn_gpu(size: usize, depth: usize, seconds: f32, ep: &str) -> Result<()> {
    let provider = ExecutionProvider::parse(ep).context("unknown --ep")?;
    let mut load = lithing_tracking::stress::GpuLoad::new(size, depth, provider, 0)?;
    let end = Instant::now() + Duration::from_secs_f32(seconds);
    let mut runs = 0u64;
    let start = Instant::now();
    while Instant::now() < end {
        load.run()?;
        runs += 1;
    }
    let tflops = runs as f64 * 2.0 * depth as f64 * (size as f64).powi(3) / start.elapsed().as_secs_f64() / 1e12;
    eprintln!("burn-gpu: {runs} runs, {tflops:.2} TFLOP/s");
    Ok(())
}

#[allow(dead_code)]
fn print_summary(summary: &Summary) {
    println!("{}", summary.table_row());
}
