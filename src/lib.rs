mod layers;
mod random;
mod onehot;

use custos::{Matrix, Device};
pub use random::*;
pub use layers::*;
pub use onehot::*;

pub trait NeuralNetwork<T> {
    fn forward(&mut self, inputs: Matrix<T>) -> Matrix<T>;
    fn backward(&mut self, grad: Matrix<T>) -> Matrix<T>;
}

/*pub struct Network<T> {
    layers: Vec<Box<dyn Layer<T>>>,
}*/


pub fn create_sine<D: Device<f32>>(device: &D, min: usize, max: usize) -> (Matrix<f32>, Matrix<f32>) {
    let mut x: Vec<f32> = Vec::new(); 
    let mut add = 0f32;
    for _ in min..max {
        x.push(add/1000.);
        add += 1.
    }

    let y = x.iter().map(|v| (v).sin()).collect::<Vec<f32>>();
    let x = Matrix::from((device, (max-min, 1), x));
    let y = Matrix::from((device, (max-min, 1), y));
    (x, y)
}