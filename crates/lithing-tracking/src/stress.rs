//! Synthetic GPU load for contention benchmarks: a chain of large matrix
//! products built in memory with ONNX Runtime's model editor (no file), run
//! in a loop on the GPU provider like a game's compute work would.

use crate::inference::{ExecutionProvider, InferenceError, Result};
use ort::editor::{Graph, Model, Node, ONNX_DOMAIN, Opset};
use ort::value::Outlet;
use ort::memory::Allocator;
use ort::session::Session;
use ort::value::{Shape, SymbolicDimensions, Tensor, TensorElementType, ValueType};

pub struct GpuLoad {
    session: Session,
    input: Tensor<f32>,
}

impl GpuLoad {
    /// `size`x`size` matrices, `depth` products per run (2 * depth * size^3 flops).
    pub fn new(size: usize, depth: usize, provider: ExecutionProvider, device: i32) -> Result<Self> {
        let n = size as i64;
        let matrix = || ValueType::Tensor { ty: TensorElementType::Float32, shape: Shape::new([n, n]), dimension_symbols: SymbolicDimensions::empty(2) };
        let mut graph = Graph::new()?;
        graph.set_inputs([Outlet::new("x", matrix())])?;
        graph.set_outputs([Outlet::new(format!("y{depth}"), matrix())])?;
        let mut weight = Tensor::<f32>::new(&Allocator::default(), [n, n])?;
        {
            let (_, data) = weight.extract_tensor_mut();
            // A scaled identity keeps the values finite however deep the chain.
            for i in 0..size {
                data[i * size + i] = 1.0;
            }
        }
        graph.add_initializer("w", weight, false)?;
        for k in 1..=depth {
            let input = if k == 1 { "x".to_string() } else { format!("y{}", k - 1) };
            graph.add_node(Node::new("MatMul", ONNX_DOMAIN, format!("mm{k}"), [input.as_str(), "w"], [format!("y{k}")], [])?)?;
        }
        let mut model = Model::new([Opset::new(ONNX_DOMAIN, 17)?])?;
        model.set_graph(graph)?;
        let env = crate::inference::environment()?;
        let mut builder = Session::builder(env)?;
        match provider {
            #[cfg(all(windows, feature = "directml"))]
            ExecutionProvider::DirectML => {
                builder = builder.with_memory_pattern(false)?.with_parallel_execution(false)?.with_execution_providers([ort::ep::DirectML::default().with_device_id(device).build().error_on_failure()])?;
            }
            #[cfg(feature = "cuda")]
            ExecutionProvider::Cuda => {
                builder = builder.with_execution_providers([ort::ep::CUDA::default().with_device_id(device).build().error_on_failure()])?;
            }
            ExecutionProvider::Cpu => {}
            _ => return Err(InferenceError::Unavailable(provider.name())),
        }
        let _ = device;
        let session = model.into_session(&builder)?;
        let mut input = Tensor::<f32>::new(&Allocator::default(), [n, n])?;
        input.extract_tensor_mut().1.fill(0.5);
        Ok(Self { session, input })
    }

    pub fn run(&mut self) -> Result<()> {
        self.session.run(ort::inputs![&self.input])?;
        Ok(())
    }
}
