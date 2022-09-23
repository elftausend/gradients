mod config;
mod init;

pub use config::*;
pub use init::{Glorot, RandomUniform};

use custos::{number::Float, Alloc, CDatatype, GenericBlas, GraphReturn};
use custos_math::{CudaTranspose, Matrix};

use crate::{GetParam, Param, WithDevice};

type LinearParams<'a, T> = (Matrix<'a, T>, Option<Matrix<'a, T>>);

pub struct Linear<'a, T, const I: usize, const O: usize> {
    pub weights: Matrix<'a, T>,
    pub bias: Option<Matrix<'a, T>>,
    pub dweights: Option<Matrix<'a, T>>,
    pub dbias: Option<Matrix<'a, T>>,
    inputs: Option<Matrix<'a, T>>,
    pub l2_reg: T,
}

impl<'a, T: Copy + Float, const I: usize, const O: usize> Linear<'a, T, I, O> {
    pub fn new<'b: 'a, D>(args: impl LinearArgs<'a, T, D, I, O>,) -> Linear<'a, T, I, O> 
    where 
        D: Alloc<T> + GraphReturn + 'a
    {
        let args = args.linear_arg();

        let config = args.config;
        let device = args.device;

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
    fn with<'b: 'a, D: Alloc<T> + GraphReturn>(device: &'b D) -> Self
    where
        Self: Default,
    {
        Self::new(device)
    }
}

impl<'a, T: Float + GenericBlas + CDatatype, const I: usize, const O: usize> Linear<'a, T, I, O> {
    pub fn forward(&mut self, inputs: &Matrix<'a, T>) -> Matrix<'a, T> {
        self.inputs = Some(inputs.shallow_or_clone());
        let mut forward = inputs.gemm(&self.weights);
        
        if let Some(bias) = &self.bias {
            forward.add_row_mut(bias);
        }
        
        forward
    }

    pub fn backward(&mut self, grad: &Matrix<'a, T>) -> Matrix<'a, T>
    where
        T: CudaTranspose,
    {
        self.dbias = if self.bias.is_some() {
            Some(grad.sum_rows())
        } else {
            None
        };
        
        self.dweights = Some(self.inputs.as_ref().unwrap().T().gemm(grad));
        grad.gemm(&self.weights.T())
    }
}

impl<'a, T: Copy, const I: usize, const O: usize> GetParam<'a, T> for Linear<'a, T, I, O> {
    fn params(&mut self) -> Option<Param<'a, T>> {
        Some(Param::new(
            self.weights.shallow(),
            self.bias.as_ref().map(|bias| bias.shallow()),
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

#[cfg(test)]
mod tests {
    use crate::{Linear, LinearConfig, Glorot};
    use custos::CPU;

    #[test]
    fn test_bias() {
        let device = CPU::new();

        let linear = Linear::<f32, 8, 16>::new((&device, LinearConfig {
            init: Glorot::new(),
            bias: false,
            ..Default::default()
        }));

        assert!(linear.bias.is_none());
    }
}
