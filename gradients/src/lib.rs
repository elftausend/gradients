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

pub struct Params<'a, T, D: Device> {
    weights: &'a mut Matrix<'a, T, D>,
    bias: Option<&'a mut Matrix<'a, T, D>>,
    dweights: &'a Matrix<'a, T, D>,
    dbias: Option<&'a Matrix<'a, T, D>>,
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

pub trait NeuralNetwork<'a, T, D: Device> {
    fn forward(&mut self, inputs: &Matrix<'a, T, D>) -> Matrix<'a, T, D>;
    fn backward(&mut self, grad: &Matrix<'a, T, D>) -> Matrix<'a, T, D>;
    fn params(&mut self) -> Vec<Param<'a, T, D>>;
}

pub fn create_sine<'a, D: Alloc<'a, f32> + IsShapeIndep>(
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

pub fn create_line<'a, T: Float, D: Alloc<'a, T> + IsShapeIndep>(
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

pub trait Bounds<'a, T>:
    CloneBuf<'a, T>
    + Gemm<T>
    + RowOp<T>
    + BaseOps<T>
    + SumOps<T>
    + TransposeOp<T>
    + AdditionalOps<T>
    + AssignOps<T>
    + Alloc<'a, T>
    + RandOp<T>
    + nn::ActivationOps<T>
    + nn::SoftmaxOps<T>
    + SumOverOps<T>
{
}

impl<
        'a,
        T: custos::number::Number + GenericBlas,
        D: CloneBuf<'a, T>
            + Gemm<T>
            + RowOp<T>
            + BaseOps<T>
            + SumOps<T>
            + TransposeOp<T>
            + AdditionalOps<T>
            + AssignOps<T>
            + Alloc<'a, T>
            + RandOp<T>
            + nn::ActivationOps<T>
            + nn::SoftmaxOps<T>
            + SumOverOps<T>,
    > Bounds<'a, T> for D
{
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
