mod accuracy;
mod batch;
mod layers;
mod ml;
mod onehot;
mod opt;

//exports of dependencies
use custos::number::Float;
pub use custos::*;
pub use custos_math::*;
pub mod purpur {
    pub use purpur::*;
}
pub use gradients_derive::*;

pub use accuracy::*;
pub use batch::*;
pub use layers::*;
pub use ml::*;
pub use onehot::*;
pub use opt::*;

pub trait GetParam<'a, T, D: Device = CPU> {
    fn params(&mut self) -> Option<Param<'a, T, D>> {
        None
    }
}

pub trait WithDevice<'a, T, D: Device = CPU> {
    fn with<'b: 'a>(_device: &'b D) -> Self
    where
        Self: Default,
    {
        Self::default()
    }
}

pub struct Param<'a, T, D: Device = CPU> {
    pub weights: Matrix<'a, T, D>,
    pub bias: Option<Matrix<'a, T, D>>,
    pub dweights: Matrix<'a, T, D>,
    pub dbias: Matrix<'a, T, D>,
}

impl<'a, T, D: Device> Param<'a, T, D> {
    pub fn new(
        weights: Matrix<'a, T, D>,
        bias: Option<Matrix<'a, T, D>>,
        dweights: Matrix<'a, T, D>,
        dbias: Matrix<'a, T, D>,
    ) -> Param<'a, T, D> {
        Param {
            weights,
            bias,
            dweights,
            dbias,
        }
    }
}

pub trait NeuralNetwork<'a, T> {
    fn forward(&mut self, inputs: &Matrix<'a, T>) -> Matrix<'a, T>;
    fn backward(&mut self, grad: &Matrix<'a, T>) -> Matrix<'a, T>;
    fn params(&mut self) -> Vec<Param<'a, T>>;
}

pub fn create_sine<'a, D: Alloc<'a, f32> + GraphReturn>(
    device: &'a D,
    min: usize,
    max: usize,
) -> (Matrix<f32, D>, Matrix<f32, D>) {
    let mut x: Vec<f32> = Vec::new();
    for add in min..max {
        x.push(add as f32 / 1000.);
    }
    let y = x
        .iter()
        .map(|v| (2. * v * std::f32::consts::PI).sin())
        .collect::<Vec<f32>>();
    let x = Matrix::from((device, (max - min, 1), x));
    let y = Matrix::from((device, (max - min, 1), y));
    (x, y)
}

pub fn create_line<'a, T: Float, D: Alloc<'a, T> + GraphReturn>(
    device: &'a D,
    min: usize,
    max: usize,
) -> (Matrix<T, D>, Matrix<T, D>) {
    let mut x: Vec<T> = Vec::with_capacity(max - min);
    for add in min..max {
        x.push(T::from_usize(add) / T::from_usize(1000));
    }
    let y = x.iter().map(|v| T::two() * *v).collect::<Vec<T>>();
    let x = Matrix::from((device, (max - min, 1), x));
    let y = Matrix::from((device, (max - min, 1), y));
    (x, y)
}

pub mod prelude {
    pub use crate::{
        correct_classes, linear::*, network, nn::*, range, Adam, Batch, LinearReg, Matrix,
        OneHotMat, OnehotOp, PolynomialReg, ReLU, Softmax, Tanh, WithDevice, CPU, SGD,
    };
    pub use purpur::*;

    #[cfg(feature = "opencl")]
    pub use crate::OpenCL;
}
