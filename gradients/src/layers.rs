use crate::{GetParam, Param, RandMatrix};
use custos::{cpu::TBlas, number::Float, GenericOCL, Matrix};
use custos_math::{
    nn::{Activations, Softmax as TSoftmax},
    Additional, Row, Sum, Transpose,
};

#[derive(Clone)]
pub struct Linear<T> {
    pub weights: Matrix<T>,
    pub bias: Matrix<T>,
    pub dweights: Option<Matrix<T>>,
    pub dbias: Option<Matrix<T>>,
    inputs: Option<Matrix<T>>,
}

impl<T: Float + TBlas + GenericOCL> Linear<T> {
    pub fn new(input_size: usize, output_size: usize) -> Linear<T> {
        let mut weights = Matrix::<T>::from((input_size, output_size));

        let glorot = (T::from_usize(6) / T::from_usize(input_size + output_size)).sqrt();
        weights.rand(glorot.negate(), glorot);

        //let weights = weights.muls(weight_size);
        //let weights = weights + (T::one() / T::from_usize(100));

        let bias = Matrix::<T>::from((1, output_size));

        Linear {
            weights,
            bias,
            dweights: None,
            dbias: None,
            inputs: None,
        }
    }

    pub fn forward(&mut self, inputs: Matrix<T>) -> Matrix<T> {
        self.inputs = Some(inputs);
        inputs.gemm(&self.weights).add_row(self.bias)
    }

    pub fn backward(&mut self, grad: Matrix<T>) -> Matrix<T> {
        self.dbias = Some(grad.sum_rows());
        self.dweights = Some(self.inputs.unwrap().T().gemm(&grad));
        grad.gemm(&self.weights.T())
    }

    pub fn sgd(&mut self, lr: T) {
        let dweights = self.dweights.unwrap();
        let dbias = self.dbias.unwrap();

        self.weights -= &dweights.muls(lr);
        self.bias -= &dbias.muls(lr);

        /*
        for (idx, value) in self.weights.as_mut_slice().iter_mut().enumerate() {
            *value -= dweights.as_slice()[idx] * lr;
        }

        for (idx, value) in self.bias.as_mut_slice().iter_mut().enumerate() {
            *value -= dbias.as_slice()[idx] * lr;
        }
        */
    }
}

impl<T: Copy> GetParam<T> for Linear<T> {
    fn get_params(&self) -> Option<Param<T>> {
        Some(Param::new(
            self.weights,
            self.bias,
            self.dweights.unwrap(),
            self.dbias.unwrap(),
        ))
    }
}

impl<T> Default for Linear<T> {
    fn default() -> Self {
        Self {
            weights: Default::default(),
            bias: Default::default(),
            dweights: Default::default(),
            dbias: Default::default(),
            inputs: Default::default(),
        }
    }
}

#[derive(Clone)]
pub struct ReLU<T> {
    inputs: Option<Matrix<T>>,
}

impl<T: Float + GenericOCL> ReLU<T> {
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

impl<T: Copy> GetParam<T> for ReLU<T> {
    fn get_params(&self) -> Option<Param<T>> {
        None
    }
}
impl<T> Default for ReLU<T> {
    fn default() -> Self {
        Self {
            inputs: Default::default(),
        }
    }
}

pub struct Softmax<T> {
    activated: Option<Matrix<T>>,
}

impl<T: GenericOCL + TBlas> Softmax<T> {
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

impl<T: Copy> GetParam<T> for Softmax<T> {
    fn get_params(&self) -> Option<Param<T>> {
        None
    }
}

impl<T> Default for Softmax<T> {
    fn default() -> Self {
        Self {
            activated: Default::default(),
        }
    }
}

/*
#[derive(Clone, Copy)]
pub struct Tanh<T> {
    inputs: Option<Matrix<T>>
}

impl <T: Float>Tanh<T> {
    pub fn new() -> Tanh<T> {
        Tanh {
            inputs: None,
        }
    }
    pub fn forward(&mut self, inputs: Matrix<T>) -> Matrix<T> {
        self.inputs = Some(inputs);
        inputs.tanh()
    }
    pub fn backward(&self, grad: Matrix<T>) -> Matrix<T> {
        self.inputs.unwrap().tanh_grad() * grad
    }
}
*/
