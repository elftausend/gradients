mod init;
mod config;

pub use init::{Glorot, RandomUniform};
pub use config::*;

use custos::{number::Float, Alloc, CDatatype, GenericBlas, GraphReturn};
use custos_math::{CudaTranspose, Matrix};

use crate::{GetParam, Param, WithDevice};


pub struct 
Linear<'a, T, const I: usize, const O: usize> {
    pub weights: Matrix<'a, T>,
    pub bias: Matrix<'a, T>,
    pub dweights: Option<Matrix<'a, T>>,
    pub dbias: Option<Matrix<'a, T>>,
    inputs: Option<Matrix<'a, T>>,
    pub l2_reg: T,
}

impl<'a, T: Copy + Float, const I: usize, const O: usize> Linear<'a, T, I, O> {
    pub fn new<'b: 'a, D: Alloc<T> + GraphReturn>(device: &'b D, config: impl AsLinearConfig<'a, T, D, I, O>) -> Linear<'a, T, I, O> {
        let config = config.as_linear_config();

        let (weights, bias) = config.init_params(device);

        Linear {
            weights,
            bias,
            dweights: None,
            dbias: None,
            inputs: None,
            l2_reg: T::zero(),
        }
    }
}

impl<'a, T: Copy + Float, const I: usize, const O: usize> WithDevice<'a, T>
    for Linear<'a, T, I, O>
{
    fn with_device<'b: 'a, D: Alloc<T> + GraphReturn>(device: &'b D) -> Self
    where
        Self: Default,
    {
        Self::new(device, ())
    }
}

impl<'a, T: Float + GenericBlas + CDatatype, const I: usize, const O: usize> Linear<'a, T, I, O> {
    pub fn forward(&mut self, inputs: &Matrix<'a, T>) -> Matrix<'a, T> {
        self.inputs = Some(inputs.shallow_or_clone());
        let mut forward = inputs.gemm(&self.weights);
        forward.add_row_mut(&self.bias);
        forward
    }

    pub fn backward(&mut self, grad: &Matrix<'a, T>) -> Matrix<'a, T>
    where
        T: CudaTranspose,
    {
        self.dbias = Some(grad.sum_rows());
        self.dweights = Some(self.inputs.as_ref().unwrap().T().gemm(grad));
        grad.gemm(&self.weights.T())
    }

    pub fn sgd(&mut self, lr: T) {
        let dweights = self.dweights.as_ref().unwrap();
        let dbias = self.dbias.as_ref().unwrap();

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

impl<'a, T: Copy, const I: usize, const O: usize> GetParam<'a, T> for Linear<'a, T, I, O> {
    fn params(&mut self) -> Option<Param<'a, T>> {
        Some(Param::new(
            self.weights.shallow(),
            self.bias.shallow(),
            self.dweights.as_ref().unwrap().shallow(),
            self.dbias.as_ref().unwrap().shallow(),
        ))
    }
}

impl<'a, T: Default, const I: usize, const O: usize> Default for Linear<'a, T, I, O> {
    fn default() -> Self {
        Self {
            weights: Default::default(),
            bias: Default::default(),
            dweights: Default::default(),
            dbias: Default::default(),
            inputs: Default::default(),
            l2_reg: Default::default(),
        }
    }
}
