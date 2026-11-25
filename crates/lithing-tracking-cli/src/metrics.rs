//! Per-run measurements: model times, detection rates, capture-to-result
//! latency, process CPU and memory (sysinfo), GPU utilisation (nvidia-smi
//! when present).

use crate::TrackArgs;
use lithing_tracking::pipeline::TrackerInfo;
use lithing_tracking::{Tracker, TrackingFrame};
use serde::{Deserialize, Serialize};
use std::io::BufRead;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Stat {
    pub n: usize,
    pub mean: f32,
    pub p50: f32,
    pub p95: f32,
    pub max: f32,
}

impl Stat {
    pub fn of(values: &mut [f32]) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        values.sort_by(f32::total_cmp);
        let at = |q: f32| values[((values.len() - 1) as f32 * q).round() as usize];
        Self { n: values.len(), mean: values.iter().sum::<f32>() / values.len() as f32, p50: at(0.5), p95: at(0.95), max: values[values.len() - 1] }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Summary {
    pub label: String,
    pub provider: String,
    pub face_providers: Vec<String>,
    pub hand_providers: Vec<String>,
    pub source: String,
    pub width: u32,
    pub height: u32,
    pub source_fps: f32,
    pub seconds: f32,
    pub captured: u64,
    pub processed: u64,
    pub dropped: u64,
    pub face_runs_per_s: f32,
    pub face_found_per_s: f32,
    pub hand_runs_per_s: f32,
    pub hands_per_run: f32,
    pub face_ms: Stat,
    pub face_detector_runs: u64,
    pub hands_ms: Stat,
    pub palm_runs: u64,
    pub latency_ms: Stat,
    /// Process CPU, in percent of one logical CPU (100 = one core busy).
    pub cpu_percent: f32,
    /// The same, as a share of the whole machine.
    pub cpu_machine_percent: f32,
    pub ram_mb_peak: f32,
    pub gpu_percent: Option<f32>,
    pub gpu_memory_mb: Option<f32>,
    pub load_ms: f32,
    pub errors: u64,
    /// Times the adaptive priority raised the inference threads to normal.
    pub priority_boosts: u64,
    pub settings: String,
}

impl Summary {
    pub fn table_header(&self) -> String {
        "label | ep (face/hands) | face det/s | face found/s | hand det/s | face ms p50/p95 | hands ms p50/p95 | latency ms p50/p95 | dropped | CPU % (1 core=100) | RAM MB | GPU %".into()
    }
    pub fn table_row(&self) -> String {
        format!(
            "{} | {}/{} | {:.1} | {:.1} | {:.1} | {:.2}/{:.2} | {:.2}/{:.2} | {:.1}/{:.1} | {:.1}% | {:.0} | {:.0} | {}",
            self.label,
            self.face_providers.join("+"),
            self.hand_providers.join("+"),
            self.face_runs_per_s,
            self.face_found_per_s,
            self.hand_runs_per_s,
            self.face_ms.p50,
            self.face_ms.p95,
            self.hands_ms.p50,
            self.hands_ms.p95,
            self.latency_ms.p50,
            self.latency_ms.p95,
            if self.captured > 0 { self.dropped as f32 * 100.0 / self.captured as f32 } else { 0.0 },
            self.cpu_percent,
            self.ram_mb_peak,
            self.gpu_percent.map(|g| format!("{g:.0}")).unwrap_or_else(|| "-".into())
        )
    }
}

struct Samples {
    cpu: Vec<f32>,
    ram_peak: f32,
    gpu: Vec<f32>,
    gpu_memory: Vec<f32>,
}

pub struct Collector {
    summary: Summary,
    start: Instant,
    face_ms: Vec<f32>,
    hands_ms: Vec<f32>,
    latency: Vec<f32>,
    face_runs: u64,
    face_found: u64,
    hand_runs: u64,
    hands_found: u64,
    window: (Instant, u64, u64, u64),
    samples: Arc<Mutex<Samples>>,
    stop: Arc<AtomicBool>,
    sampler: Option<std::thread::JoinHandle<()>>,
    nvidia: Option<Child>,
}

impl Collector {
    pub fn start(label: &str, args: &TrackArgs, info: &TrackerInfo) -> Self {
        let summary = Summary {
            label: label.into(),
            provider: args.ep.clone(),
            face_providers: info.face_providers.iter().map(|p| p.name().to_string()).collect(),
            hand_providers: info.hand_providers.iter().map(|p| p.name().to_string()).collect(),
            source: info.source.description.clone(),
            width: info.source.width,
            height: info.source.height,
            source_fps: info.source.fps,
            load_ms: info.load_ms,
            settings: format!(
                "face {} fps, hands {} fps{}, threads {}, priority {}{}, palm interval {}{}{}{}",
                args.face_fps,
                args.hand_fps,
                if args.no_hands { " (off)" } else { "" },
                args.threads,
                args.priority,
                if args.fixed_priority { " (fixed)" } else { " (adaptive)" },
                args.palm_interval,
                if args.serial { ", serial" } else { ", parallel" },
                args.affinity.map(|a| format!(", {a} CPUs")).unwrap_or_default(),
                args.modes.describe()
            ),
            ..Default::default()
        };
        let samples = Arc::new(Mutex::new(Samples { cpu: Vec::new(), ram_peak: 0.0, gpu: Vec::new(), gpu_memory: Vec::new() }));
        let stop = Arc::new(AtomicBool::new(false));
        // nvidia-smi samples total GPU use twice a second, when there is one.
        let mut nvidia = Command::new("nvidia-smi")
            .args(["--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-lms", "500"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok();
        if let Some(child) = &mut nvidia
            && let Some(stdout) = child.stdout.take()
        {
            let samples = samples.clone();
            std::thread::spawn(move || {
                for line in std::io::BufReader::new(stdout).lines().map_while(|l| l.ok()) {
                    let mut parts = line.split(',').map(|p| p.trim().parse::<f32>());
                    if let (Some(Ok(util)), Some(Ok(memory))) = (parts.next(), parts.next()) {
                        let mut s = samples.lock().unwrap();
                        s.gpu.push(util);
                        s.gpu_memory.push(memory);
                    }
                }
            });
        }
        let sampler = {
            let (samples, stop) = (samples.clone(), stop.clone());
            std::thread::spawn(move || {
                use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System};
                let pid = sysinfo::get_current_pid().ok();
                let mut system = System::new();
                let kind = ProcessRefreshKind::nothing().with_cpu().with_memory();
                let refresh = |system: &mut System| {
                    if let Some(pid) = pid {
                        system.refresh_processes_specifics(ProcessesToUpdate::Some(&[pid]), true, kind);
                    }
                };
                refresh(&mut system);
                while !stop.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(500));
                    refresh(&mut system);
                    if let Some(process) = pid.and_then(|pid| system.process(pid)) {
                        let mut s = samples.lock().unwrap();
                        s.cpu.push(process.cpu_usage());
                        s.ram_peak = s.ram_peak.max(process.memory() as f32 / 1048576.0);
                    }
                }
            })
        };
        Self {
            summary,
            start: Instant::now(),
            face_ms: Vec::new(),
            hands_ms: Vec::new(),
            latency: Vec::new(),
            face_runs: 0,
            face_found: 0,
            hand_runs: 0,
            hands_found: 0,
            window: (Instant::now(), 0, 0, 0),
            samples,
            stop,
            sampler: Some(sampler),
            nvidia,
        }
    }

    pub fn add(&mut self, frame: &TrackingFrame) {
        self.latency.push(frame.timings.latency_ms);
        if frame.face_updated {
            self.face_runs += 1;
            self.face_ms.push(frame.timings.face_ms);
            if frame.face.is_some() {
                self.face_found += 1;
            }
            if frame.face.as_ref().is_some_and(|f| f.detected) {
                self.summary.face_detector_runs += 1;
            }
        }
        if frame.hands_updated {
            self.hand_runs += 1;
            self.hands_ms.push(frame.timings.hands_ms);
            self.hands_found += frame.raw_hands.len() as u64;
            if frame.timings.palm_detector_ms > 0.0 {
                self.summary.palm_runs += 1;
            }
        }
    }

    pub fn progress_line(&mut self) -> String {
        let seconds = self.window.0.elapsed().as_secs_f32();
        let line = format!(
            "{:5.1}s  face {:4.1}/s (found {:4.1}/s)  hands {:4.1}/s  face {:.2} ms  hands {:.2} ms  latency {:.1} ms",
            self.start.elapsed().as_secs_f32(),
            (self.face_runs - self.window.1) as f32 / seconds,
            (self.face_found - self.window.3) as f32 / seconds,
            (self.hand_runs - self.window.2) as f32 / seconds,
            self.face_ms.last().copied().unwrap_or(0.0),
            self.hands_ms.last().copied().unwrap_or(0.0),
            self.latency.last().copied().unwrap_or(0.0),
        );
        self.window = (Instant::now(), self.face_runs, self.hand_runs, self.face_found);
        line
    }

    pub fn finish(mut self, tracker: &Tracker) -> Summary {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(sampler) = self.sampler.take() {
            let _ = sampler.join();
        }
        if let Some(mut child) = self.nvidia.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        let seconds = self.start.elapsed().as_secs_f32();
        let s = &mut self.summary;
        s.seconds = seconds;
        s.captured = tracker.stats.captured.load(Ordering::Relaxed);
        s.dropped = tracker.stats.dropped.load(Ordering::Relaxed);
        s.processed = tracker.stats.processed.load(Ordering::Relaxed);
        s.errors = tracker.stats.errors.load(Ordering::Relaxed);
        s.priority_boosts = tracker.stats.boosts.load(Ordering::Relaxed);
        s.face_runs_per_s = self.face_runs as f32 / seconds;
        s.face_found_per_s = self.face_found as f32 / seconds;
        s.hand_runs_per_s = self.hand_runs as f32 / seconds;
        s.hands_per_run = if self.hand_runs > 0 { self.hands_found as f32 / self.hand_runs as f32 } else { 0.0 };
        s.face_ms = Stat::of(&mut self.face_ms);
        s.hands_ms = Stat::of(&mut self.hands_ms);
        s.latency_ms = Stat::of(&mut self.latency);
        let samples = self.samples.lock().unwrap();
        // Skip the first sample (model loading is over by then, but warm-up is not).
        let cpu: Vec<f32> = samples.cpu.iter().skip(1).copied().collect();
        s.cpu_percent = if cpu.is_empty() { 0.0 } else { cpu.iter().sum::<f32>() / cpu.len() as f32 };
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) as f32;
        s.cpu_machine_percent = s.cpu_percent / cores;
        s.ram_mb_peak = samples.ram_peak;
        if !samples.gpu.is_empty() {
            s.gpu_percent = Some(samples.gpu.iter().sum::<f32>() / samples.gpu.len() as f32);
            s.gpu_memory_mb = samples.gpu_memory.iter().copied().reduce(f32::max);
        }
        if let Some(error) = tracker.stats.last_error.lock().unwrap().clone() {
            eprintln!("last error: {error}");
        }
        self.summary.clone()
    }
}
