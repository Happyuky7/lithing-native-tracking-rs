//! ONNX Runtime sessions, execution providers and the Auto fallback.

use crate::priority::{self, ThreadPriority};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;
use std::path::Path;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Where models run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum ExecutionProvider {
    /// GPU when it is faster, CPU when not, re-measured while running so a
    /// game that saturates the GPU (or the CPU) moves tracking to the other.
    #[default]
    Auto,
    /// DirectML (Windows, DirectX 12: NVIDIA, AMD, Intel).
    DirectML,
    /// CUDA (NVIDIA; needs the `cuda` feature and CUDA/cuDNN installed).
    Cuda,
    /// ONNX Runtime's CPU kernels (MLAS; XNNPACK with the `xnnpack` feature).
    Cpu,
}

impl ExecutionProvider {
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "directml" | "dml" | "gpu" => Some(Self::DirectML),
            "cuda" => Some(Self::Cuda),
            "cpu" => Some(Self::Cpu),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::DirectML => "directml",
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        }
    }

    /// The GPU provider Auto uses on this build, if any.
    pub fn default_gpu() -> Option<Self> {
        if cfg!(feature = "cuda") {
            Some(Self::Cuda)
        } else if cfg!(all(windows, feature = "directml")) {
            Some(Self::DirectML)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, rename_all = "camelCase"))]
pub struct InferenceOptions {
    pub provider: ExecutionProvider,
    /// Threads per CPU session. 1 runs every operator on the calling worker
    /// thread (cheapest, and it keeps the worker's priority).
    pub cpu_threads: usize,
    pub gpu_device: i32,
    /// Priority of the inference threads; below normal keeps a game ahead.
    pub priority: ThreadPriority,
    /// How often Auto times the other provider once (seconds).
    pub auto_probe_seconds: f32,
}

impl Default for InferenceOptions {
    fn default() -> Self {
        Self { provider: ExecutionProvider::Auto, cpu_threads: 1, gpu_device: 0, priority: ThreadPriority::BelowNormal, auto_probe_seconds: 4.0 }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("ONNX Runtime: {0}")]
    Ort(String),
    #[error("model file {0}: {1}")]
    Model(String, String),
    #[error("execution provider {0} is not available in this build or on this machine")]
    Unavailable(&'static str),
}

impl<T> From<ort::Error<T>> for InferenceError {
    fn from(error: ort::Error<T>) -> Self {
        Self::Ort(error.to_string())
    }
}

pub type Result<T> = std::result::Result<T, InferenceError>;

pub(crate) fn environment() -> Result<&'static ort::environment::Environment> {
    static ENV: OnceLock<std::result::Result<ort::environment::Environment, String>> = OnceLock::new();
    ENV.get_or_init(|| ort::init().with_name("lithing-tracking").with_telemetry(false).build().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| InferenceError::Ort(e.clone()))
}

/// Spawns ONNX Runtime's pool threads at the configured priority.
struct PriorityThreads(ThreadPriority);

impl ort::environment::ThreadManager for PriorityThreads {
    type Thread = std::thread::JoinHandle<()>;
    fn create(&self, work: impl FnOnce() + Send + 'static) -> ort::Result<Self::Thread> {
        let priority = self.0;
        std::thread::Builder::new()
            .name("ort-pool".into())
            .spawn(move || {
                priority::set_current(priority);
                work();
            })
            .map_err(|e| ort::Error::new(e.to_string()))
    }
    fn join(thread: Self::Thread) -> ort::Result<()> {
        let _ = thread.join();
        Ok(())
    }
}

/// One loaded model on one provider.
pub struct Model {
    session: Session,
    pub provider: ExecutionProvider,
    pub name: String,
}

impl Model {
    pub fn load(path: &Path, provider: ExecutionProvider, options: &InferenceOptions) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| InferenceError::Model(path.display().to_string(), e.to_string()))?;
        let env = environment()?;
        let mut builder = Session::builder(env)?.with_optimization_level(GraphOptimizationLevel::All)?;
        if std::env::var_os("LITHING_ORT_VERBOSE").is_some() {
            // Prints ONNX Runtime's node placement (which provider runs each operator).
            env.set_log_level(ort::logging::LogLevel::Verbose);
            builder = builder.with_log_level(ort::logging::LogLevel::Verbose)?;
        }
        match provider {
            ExecutionProvider::Cpu | ExecutionProvider::Auto => {
                // No spinning: an idle pool thread must not burn a core the game wants.
                builder = builder.with_intra_threads(options.cpu_threads.max(1))?.with_inter_threads(1)?.with_intra_op_spinning(false)?.with_inter_op_spinning(false)?;
                if options.cpu_threads > 1 {
                    builder = builder.with_thread_manager(PriorityThreads(options.priority))?;
                }
                #[cfg(feature = "xnnpack")]
                {
                    use ort::ep::ExecutionProvider as _;
                    let ep = ort::ep::XNNPACK::default().with_intra_op_num_threads(std::num::NonZeroUsize::new(options.cpu_threads.max(1)).unwrap());
                    if ep.is_available().unwrap_or(false) {
                        builder = builder.with_execution_providers([ep.build()])?;
                    }
                }
            }
            ExecutionProvider::DirectML => {
                #[cfg(all(windows, feature = "directml"))]
                {
                    // DirectML needs sequential execution without memory patterns.
                    builder = builder
                        .with_memory_pattern(false)?
                        .with_parallel_execution(false)?
                        .with_intra_threads(1)?
                        .with_intra_op_spinning(false)?
                        .with_execution_providers([ort::ep::DirectML::default().with_device_id(options.gpu_device).build().error_on_failure()])?;
                }
                #[cfg(not(all(windows, feature = "directml")))]
                return Err(InferenceError::Unavailable("directml"));
            }
            ExecutionProvider::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    builder = builder.with_intra_threads(1)?.with_intra_op_spinning(false)?.with_execution_providers([ort::ep::CUDA::default().with_device_id(options.gpu_device).build().error_on_failure()])?;
                }
                #[cfg(not(feature = "cuda"))]
                return Err(InferenceError::Unavailable("cuda"));
            }
        }
        let session = builder.commit_from_memory(&bytes)?;
        let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        Ok(Self { session, provider: if provider == ExecutionProvider::Auto { ExecutionProvider::Cpu } else { provider }, name })
    }

    /// Runs the model on one NHWC/feature input and copies each output into `outputs` (in order).
    pub fn run(&mut self, input: &[f32], shape: &[usize], outputs: &mut [&mut Vec<f32>]) -> Result<()> {
        let tensor = TensorRef::from_array_view((shape.to_vec(), input))?;
        let results = self.session.run(ort::inputs![tensor])?;
        for (index, out) in outputs.iter_mut().enumerate() {
            let (_, data) = results[index].try_extract_tensor::<f32>()?;
            out.clear();
            out.extend_from_slice(data);
        }
        Ok(())
    }
}

/// A model that may run on a GPU and/or the CPU, picking the faster one.
pub struct AdaptiveModel {
    gpu: Option<Model>,
    cpu: Option<Model>,
    use_gpu: bool,
    auto: bool,
    /// Moving average run time (ms) of [gpu, cpu].
    average: [f32; 2],
    last_probe: Instant,
    probe_interval: Duration,
    pub switches: u32,
}

impl AdaptiveModel {
    pub fn load(path: &Path, options: &InferenceOptions) -> Result<Self> {
        let (gpu, cpu) = match options.provider {
            ExecutionProvider::Cpu => (None, Some(Model::load(path, ExecutionProvider::Cpu, options)?)),
            ExecutionProvider::DirectML | ExecutionProvider::Cuda => (Some(Model::load(path, options.provider, options)?), None),
            ExecutionProvider::Auto => {
                let gpu = ExecutionProvider::default_gpu().and_then(|ep| Model::load(path, ep, options).ok());
                (gpu, Some(Model::load(path, ExecutionProvider::Cpu, options)?))
            }
        };
        let auto = gpu.is_some() && cpu.is_some();
        Ok(Self { use_gpu: gpu.is_some(), gpu, cpu, auto, average: [0.0; 2], last_probe: Instant::now(), probe_interval: Duration::from_secs_f32(options.auto_probe_seconds.max(0.5)), switches: 0 })
    }

    pub fn provider(&self) -> ExecutionProvider {
        if self.use_gpu { self.gpu.as_ref().map(|m| m.provider).unwrap_or(ExecutionProvider::Cpu) } else { ExecutionProvider::Cpu }
    }

    fn model(&mut self, gpu: bool) -> &mut Model {
        if gpu { self.gpu.as_mut().or(self.cpu.as_mut()).unwrap() } else { self.cpu.as_mut().or(self.gpu.as_mut()).unwrap() }
    }

    fn timed_run(&mut self, gpu: bool, input: &[f32], shape: &[usize], outputs: &mut [&mut Vec<f32>]) -> Result<f32> {
        let start = Instant::now();
        self.model(gpu).run(input, shape, outputs)?;
        let ms = start.elapsed().as_secs_f32() * 1000.0;
        let slot = &mut self.average[if gpu { 0 } else { 1 }];
        *slot = if *slot == 0.0 { ms } else { *slot + (ms - *slot) * 0.15 };
        Ok(ms)
    }

    /// Times both providers a few times on `input` and starts on the faster.
    pub fn calibrate(&mut self, input: &[f32], shape: &[usize], outputs: &mut [&mut Vec<f32>]) -> Result<()> {
        if !self.auto {
            // Warm up the only provider (first runs compile kernels).
            for _ in 0..3 {
                let gpu = self.use_gpu;
                self.timed_run(gpu, input, shape, outputs)?;
            }
            return Ok(());
        }
        for gpu in [true, false] {
            self.timed_run(gpu, input, shape, outputs)?;
            self.average[if gpu { 0 } else { 1 }] = 0.0;
            for _ in 0..6 {
                self.timed_run(gpu, input, shape, outputs)?;
            }
        }
        // A GPU only wins clearly: a CPU run that is nearly as fast leaves the GPU to the game.
        self.use_gpu = self.average[0] * 1.25 < self.average[1];
        Ok(())
    }

    pub fn run(&mut self, input: &[f32], shape: &[usize], outputs: &mut [&mut Vec<f32>]) -> Result<()> {
        let gpu = self.use_gpu;
        self.timed_run(gpu, input, shape, outputs)?;
        if self.auto && self.last_probe.elapsed() >= self.probe_interval {
            self.last_probe = Instant::now();
            // Time the other provider once on this real input; it does not change the outputs used.
            let mut scratch: Vec<Vec<f32>> = outputs.iter().map(|o| Vec::with_capacity(o.len())).collect();
            let mut refs: Vec<&mut Vec<f32>> = scratch.iter_mut().collect();
            if let Ok(ms) = self.timed_run(!gpu, input, shape, &mut refs) {
                // One probe stands for the other provider's current speed.
                self.average[if gpu { 1 } else { 0 }] = ms;
                let (current, other) = if gpu { (self.average[0], self.average[1]) } else { (self.average[1], self.average[0]) };
                // Hysteresis: switch only for a clear win (and prefer leaving the GPU to games).
                let threshold = if gpu { 0.85 } else { 0.6 };
                if other < current * threshold {
                    self.use_gpu = !gpu;
                    self.switches += 1;
                }
            }
        }
        Ok(())
    }

    /// Moving average run time (ms) on the active provider.
    pub fn average_ms(&self) -> f32 {
        self.average[if self.use_gpu { 0 } else { 1 }]
    }
}
