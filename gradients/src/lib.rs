mod accuracy;
mod layers;
mod ml;
mod onehot;
mod opt;

//exports of dependencies
pub use custos::*;
pub use custos_math::*;
pub mod purpur {
    pub use purpur::*;
}
pub use gradients_derive::*;

pub use accuracy::*;
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
    fn with_device<D: Alloc<T>>(_device: &'a D) -> Self
    where
        Self: Default,
    {
        Self::default()
    }
}

pub struct Param<'a, T> {
    pub weights: Matrix<'a, T>,
    pub bias: Matrix<'a, T>,
    pub dweights: Matrix<'a, T>,
    pub dbias: Matrix<'a, T>,
}

impl<'a, T> Param<'a, T> {
    pub fn new(
        weights: Matrix<'a, T>,
        bias: Matrix<'a, T>,
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

pub fn create_sine<D: Alloc<f32>>(
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
