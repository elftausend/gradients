use crate::{GetParam, WithDevice};
use custos::{number::Float, CDatatype, GenericBlas};
use custos_math::Matrix;
use gradients_derive::NoParams;

#[derive(NoParams)]
pub struct ReLU<'a, T> {
    inputs: Option<Matrix<'a, T>>,
}

impl<'a, T: Float + CDatatype> ReLU<'a, T> {
    pub fn new() -> ReLU<'a, T> {
        ReLU { inputs: None }
    }
    pub fn forward(&mut self, inputs: &Matrix<'a, T>) -> Matrix<'a, T> {
        self.inputs = Some(inputs.shallow_or_clone());
        inputs.relu()
    }
    pub fn backward(&self, grad: &Matrix<'a, T>) -> Matrix<'a, T> {
        self.inputs.as_ref().unwrap().relu_grad() * grad
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

    pub fn forward(&mut self, x: &Matrix<'a, T>) -> Matrix<'a, T> {
        let activated = x.softmax();
        self.activated = Some(activated.shallow_or_clone());
        activated
    }

    pub fn backward(&self, grad: &Matrix<'a, T>) -> Matrix<'a, T> {
        grad.softmax_grad(self.activated.as_ref().unwrap())
    }
}

impl<'a, T> Default for Softmax<'a, T> {
    fn default() -> Self {
        Self {
            activated: Default::default(),
        }
    }
}

#[derive(NoParams)]
pub struct Tanh<'a, T> {
    inputs: Option<Matrix<'a, T>>,
}

impl<'a, T> Tanh<'a, T> {
    pub fn new() -> Tanh<'a, T> {
        Tanh { inputs: None }
    }
}

impl<'a, T: Float + CDatatype> Tanh<'a, T> {
    pub fn forward(&mut self, inputs: &Matrix<'a, T>) -> Matrix<'a, T> {
        self.inputs = Some(inputs.shallow_or_clone());
        inputs.tanh()
    }
    pub fn backward(&self, grad: &Matrix<'a, T>) -> Matrix<'a, T> {
        self.inputs.as_ref().unwrap().tanh_grad() * grad
    }
}

impl<'a, T> Default for Tanh<'a, T> {
    fn default() -> Self {
        Self {
            inputs: Default::default(),
        }
    }
}

/* 

#[derive(NoParams)]
pub struct Sigmoid<'a, T> {
    activated: Option<Matrix<'a, T>>,
}

impl<'a, T> Sigmoid<'a, T> {
    pub fn new() -> Sigmoid<'a, T> {
        Sigmoid::default()
    }
}

impl<'a, T: CDatatype + Float> Sigmoid<'a, T> {
    pub fn forward(&mut self, inputs: &Matrix<'a, T>) -> Matrix<'a, T> {
        let activated = inputs.sigmoid();
        self.activated = Some(inputs.shallow_or_clone());
        activated
    }

    pub fn backward(&mut self, grad: &Matrix<'a, T>) -> Matrix<'a, T> {
        self.activated.as_ref().unwrap().sigmoid_grad() * grad
    }
}

impl<'a, T> Default for Sigmoid<'a, T> {
    #[inline]
    fn default() -> Self {
        Self {
            activated: Default::default(),
        }
    }
}
*/