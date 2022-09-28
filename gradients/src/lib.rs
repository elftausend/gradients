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

pub trait GetParam<'a, T> {
    fn params(&mut self) -> Option<Param<'a, T>> {
        None
    }
}

pub trait WithDevice<'a, T> {
    fn with<'b: 'a, D: Alloc<T> + GraphReturn>(_device: &'b D) -> Self
    where
        Self: Default,
    {
        Self::default()
    }
}

pub struct Param<'a, T> {
    pub weights: Matrix<'a, T>,
    pub bias: Option<Matrix<'a, T>>,
    pub dweights: Matrix<'a, T>,
    pub dbias: Matrix<'a, T>,
}

impl<'a, T> Param<'a, T> {
    pub fn new(
        weights: Matrix<'a, T>,
        bias: Option<Matrix<'a, T>>,
        dweights: Matrix<'a, T>,
        dbias: Matrix<'a, T>,
    ) -> Param<'a, T> {
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

pub fn create_sine<D: Alloc<f32> + GraphReturn>(
    device: &D,
    min: usize,
    max: usize,
) -> (Matrix<f32>, Matrix<f32>) {
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

pub fn create_line<T: Float, D: Alloc<T> + GraphReturn>(
    device: &D,
    min: usize,
    max: usize,
) -> (Matrix<T>, Matrix<T>) {
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
        correct_classes, network, nn::*, range, Adam, Batch, Matrix, OneHotMat,
        PolynomialReg, ReLU, Softmax, Tanh, CPU, SGD, WithDevice, linear::*,
        OnehotOp, LinearReg
    };
    pub use purpur::*;

    #[cfg(feature = "opencl")]
    pub use crate::CLDevice;
}
