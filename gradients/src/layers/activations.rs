use crate::GetParam;
use custos::{number::Float, CDatatype, GenericBlas};
use custos_math::{
    nn::{Activations, Softmax as TSoftmax},
    Matrix,
};
use gradients_derive::NoParams;

#[derive(Clone, NoParams)]
pub struct ReLU<'a, T> {
    inputs: Option<Matrix<'a, T>>,
}

impl<'a, T: Float + CDatatype> ReLU<'a, T> {
    pub fn new() -> ReLU<'a, T> {
        ReLU { inputs: None }
    }
    pub fn forward(&mut self, inputs: Matrix<T>) -> Matrix<T> {
        self.inputs = Some(inputs);
        inputs.relu()
    }
    pub fn backward(&self, grad: Matrix<T>) -> Matrix<T> {
        self.inputs.unwrap().relu_grad() * grad
    }
}

impl<'a, T> Default for ReLU<'a, T> {
    fn default() -> Self {
        Self {
            inputs: Default::default(),
        }
    }
}

#[derive(NoParams)]
pub struct Softmax<'a, T> {
    activated: Option<Matrix<'a, T>>,
}

impl<'a, T: CDatatype + GenericBlas> Softmax<'a, T> {
    pub fn new() -> Self {
        Softmax { activated: None }
    }

    pub fn forward(&mut self, x: Matrix<T>) -> Matrix<T> {
        let activated = x.softmax();
        self.activated = Some(activated);
        activated
    }

    pub fn backward(&self, grad: Matrix<T>) -> Matrix<T> {
        grad.softmax_grad(&self.activated.unwrap())
    }
}

impl<'a, T> Default for Softmax<'a, T> {
    fn default() -> Self {
        Self {
            activated: Default::default(),
        }
    }
}

#[derive(Clone, NoParams)]
pub struct Tanh<'a, T> {
    inputs: Option<Matrix<'a, T>>,
}

impl<'a, T: Float + CDatatype> Tanh<'a, T> {
    pub fn new() -> Tanh<'a, T> {
        Tanh { inputs: None }
    }
    pub fn forward(&mut self, inputs: Matrix<T>) -> Matrix<T> {
        self.inputs = Some(inputs);
        inputs.tanh()
    }
    pub fn backward(&self, grad: Matrix<T>) -> Matrix<T> {
        self.inputs.unwrap().tanh_grad() * grad
    }
}

impl<'a, T> Default for Tanh<'a, T> {
    fn default() -> Self {
        Self {
            inputs: Default::default(),
        }
    }
}
