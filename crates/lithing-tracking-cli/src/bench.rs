//! The benchmark matrix: every scenario runs `lithing-track run` in a child
//! process (so a burner's CPU time never counts as tracking's), with burner
//! child processes for CPU or GPU contention, and the summaries are collected
//! into a Markdown table and a JSON file.

use crate::metrics::Summary;
use anyhow::{Context, Result};
use clap::Args;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

#[derive(Args)]
pub struct BenchArgs {
    /// Video file played like a 30 fps camera (looped).
    pub input: String,
    /// Seconds per scenario.
    #[arg(long, default_value_t = 20.0)]
    pub duration: f32,
    /// Comma list: idle, cpu-load, gpu-load, both-load, lowend, lowend-cpu-load.
    #[arg(long, default_value = "idle,cpu-load,gpu-load,both-load,lowend,lowend-cpu-load")]
    pub scenarios: String,
    /// Comma list of providers to try in each scenario (lowend always runs cpu).
    #[arg(long, default_value = "cpu,directml,auto")]
    pub eps: String,
    #[arg(long, default_value_t = 30.0)]
    pub face_fps: f32,
    #[arg(long, default_value_t = 30.0)]
    pub hand_fps: f32,
    /// Output folder for the report.
    #[arg(long, default_value = "bench-results")]
    pub out: PathBuf,
    /// Low light mode in every scenario (see `run --low-light`).
    #[arg(long)]
    pub low_light: bool,
    /// Overexposed camera mode in every scenario.
    #[arg(long)]
    pub high_light: bool,
    /// Glasses mode in every scenario.
    #[arg(long)]
    pub glasses: bool,
    /// Extra arguments passed to every `run`.
    #[arg(long, allow_hyphen_values = true)]
    pub extra: Option<String>,
}

struct Burners(Vec<Child>);

impl Drop for Burners {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

pub fn bench(args: BenchArgs) -> Result<()> {
    let exe = std::env::current_exe()?;
    std::fs::create_dir_all(&args.out)?;
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut summaries: Vec<(String, Summary)> = Vec::new();
    let burn_seconds = (args.duration + 30.0).to_string();
    for scenario in args.scenarios.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let lowend = scenario.starts_with("lowend");
        let eps: Vec<&str> = if lowend { vec!["cpu"] } else { args.eps.split(',').map(str::trim).collect() };
        for ep in eps {
            let label = format!("{scenario}/{ep}");
            let background = background_load();
            eprintln!("== {label} (background before start: {background})");
            let mut burners = Burners(Vec::new());
            let spawn = |list: &mut Vec<Child>, extra: &[&str]| -> Result<()> {
                let child = Command::new(&exe).args(extra).stdout(Stdio::null()).stderr(Stdio::null()).spawn().context("starting a burner")?;
                list.push(child);
                Ok(())
            };
            if scenario == "cpu-load" || scenario == "both-load" {
                // Every logical CPU busy at normal priority: a CPU-bound game.
                spawn(&mut burners.0, &["burn-cpu", "--threads", &cores.to_string(), "--seconds", &burn_seconds])?;
            }
            if scenario == "lowend-cpu-load" {
                spawn(&mut burners.0, &["burn-cpu", "--threads", "2", "--affinity", "2", "--seconds", &burn_seconds])?;
            }
            if scenario == "gpu-load" || scenario == "both-load" {
                spawn(&mut burners.0, &["burn-gpu", "--seconds", &burn_seconds])?;
            }
            if !burners.0.is_empty() {
                // Let the load ramp up (and the GPU clocks settle) before measuring.
                std::thread::sleep(Duration::from_secs(3));
            }
            let summary_path = args.out.join(format!("{}.json", label.replace('/', "_")));
            let duration = args.duration.to_string();
            let face_fps = args.face_fps.to_string();
            let hand_fps = args.hand_fps.to_string();
            let mut command = Command::new(&exe);
            command.args(["run", &args.input, "--loop", "--quiet", "--ep", ep, "--duration", &duration, "--face-fps", &face_fps, "--hand-fps", &hand_fps, "--label", &label]).arg("--summary-json").arg(&summary_path);
            if lowend {
                command.args(["--affinity", "2", "--threads", "1"]);
            }
            for (on, flag) in [(args.low_light, "--low-light"), (args.high_light, "--high-light"), (args.glasses, "--glasses")] {
                if on {
                    command.arg(flag);
                }
            }
            if let Some(extra) = &args.extra {
                command.args(extra.split_whitespace());
            }
            let status = command.status().context("running a scenario")?;
            drop(burners);
            if !status.success() {
                eprintln!("   {label} failed ({status})");
                continue;
            }
            let summary: Summary = serde_json::from_str(&std::fs::read_to_string(&summary_path)?)?;
            eprintln!("   {}", summary.table_row());
            summaries.push((format!("{label} [{background}]"), summary));
        }
    }
    let mut report = String::new();
    report.push_str(&format!("# lithing-tracking benchmark\n\ninput: `{}`, {} s per scenario, face {} fps, hands {} fps requested\n\n", args.input, args.duration, args.face_fps, args.hand_fps));
    if let Some((_, first)) = summaries.first() {
        report.push_str(&format!("{}\n", first.table_header()));
        report.push_str(&format!("{}\n", first.table_header().split(" | ").map(|_| "---").collect::<Vec<_>>().join(" | ")));
    }
    for (_, summary) in &summaries {
        report.push_str(&format!("{}\n", summary.table_row()));
    }
    report.push_str("
Background load just before each scenario (other programs, before any burner started):

");
    for (label, summary) in &summaries {
        report.push_str(&format!("* {label}: {} priority boosts, settings: {}
", summary.priority_boosts, summary.settings));
    }
    std::fs::write(args.out.join("report.md"), &report)?;
    std::fs::write(args.out.join("report.json"), serde_json::to_string_pretty(&summaries.iter().map(|(_, s)| s).collect::<Vec<_>>())?)?;
    println!("{report}");
    Ok(())
}

/// Machine CPU load and GPU utilisation over one second, before a scenario starts.
fn background_load() -> String {
    use sysinfo::{CpuRefreshKind, RefreshKind, System};
    let mut system = System::new_with_specifics(RefreshKind::nothing().with_cpu(CpuRefreshKind::nothing().with_cpu_usage()));
    std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL.max(Duration::from_millis(1000)));
    system.refresh_cpu_usage();
    let cpu = system.global_cpu_usage();
    let gpu = Command::new("nvidia-smi")
        .args(["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f32>().ok());
    format!("machine CPU {cpu:.0}%, GPU {}", gpu.map(|g| format!("{g:.0}%")).unwrap_or_else(|| "n/a".into()))
}
