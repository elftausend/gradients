use crate::GetParam;
use custos::{number::Float, CDatatype, GenericBlas};
use custos_math::{
    nn::{Activations, Softmax as TSoftmax},
    Matrix,
};
use gradients_derive::NoParams;

#[derive(Clone, NoParams)]
pub struct ReLU<T> {
    inputs: Option<Matrix<T>>,
}

impl<T: Float + CDatatype> ReLU<T> {
    pub fn new() -> ReLU<T> {
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

impl<T> Default for ReLU<T> {
    fn default() -> Self {
        Self {
            inputs: Default::default(),
        }
    }
}

#[derive(NoParams)]
pub struct Softmax<T> {
    activated: Option<Matrix<T>>,
}

impl<T: CDatatype + GenericBlas> Softmax<T> {
    pub fn new() -> Softmax<T> {
        Softmax { activated: None }
    }

    pub fn forward(&mut self, x: Matrix<T>) -> Matrix<T> {
        let activated = x.softmax();
        self.activated = Some(activated);
        activated
    }

    pub fn backward(&self, grad: Matrix<T>) -> Matrix<T> {
        grad.softmax_grad(self.activated.unwrap())
    }
}

impl<T> Default for Softmax<T> {
    fn default() -> Self {
        Self {
            activated: Default::default(),
        }
    }
}

#[derive(Clone, NoParams)]
pub struct Tanh<T> {
    inputs: Option<Matrix<T>>,
}

impl<T: Float + CDatatype> Tanh<T> {
    pub fn new() -> Tanh<T> {
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

impl<T> Default for Tanh<T> {
    fn default() -> Self {
        Self {
            inputs: Default::default(),
        }
    }
}
