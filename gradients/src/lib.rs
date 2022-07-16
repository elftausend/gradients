mod accuracy;
mod layers;
mod onehot;
mod opt;
mod ml;

pub use accuracy::*;
pub use layers::*;
pub use onehot::*;
pub use opt::*;
pub use ml::*;

use custos::Device;
use custos_math::Matrix;
pub use rand::distributions::uniform::SampleUniform;

pub trait GetParam<T> {
    fn params(&self) -> Option<Param<T>> {
        None
    }
}
pub struct Param<T> {
    pub weights: Matrix<T>,
    pub bias: Matrix<T>,
    pub dweights: Matrix<T>,
    pub dbias: Matrix<T>,
}

impl<T> Param<T> {
    pub fn new(
        weights: Matrix<T>,
        bias: Matrix<T>,
        dweights: Matrix<T>,
        dbias: Matrix<T>,
    ) -> Param<T> {
        Param { weights, bias, dweights, dbias }
    }
}

pub trait NeuralNetwork<T> {
    fn forward(&mut self, inputs: Matrix<T>) -> Matrix<T>;
    fn backward(&mut self, grad: Matrix<T>) -> Matrix<T>;
    fn params(&mut self) -> Vec<Param<T>>;
}

pub fn create_sine<D: Device<f32>>(
    device: &D,
    min: usize,
    max: usize,
) -> (Matrix<f32>, Matrix<f32>) {
    let mut x: Vec<f32> = Vec::new();
    for add in min..max {
        x.push(add as f32 / 1000.);
    }
    let y = x.iter().map(|v| (2. * v * std::f32::consts::PI).sin()).collect::<Vec<f32>>();
    let x = Matrix::from((device, (max - min, 1), x));
    let y = Matrix::from((device, (max - min, 1), y));
    (x, y)
}
