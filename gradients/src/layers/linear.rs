mod config;
mod init;
mod l2_reg;

use std::cell::RefCell;

pub use config::*;
pub use l2_reg::*;
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
    pub l2_reg_loss: Option<&'a RefCell<T>>
}

impl<'a, T: Copy + Float, const I: usize, const O: usize> Linear<'a, T, I, O> {
    pub fn new<'b: 'a, D>(device: &'b D, args: impl IntoLinearConfig<'a, T, D, I, O>,) -> Linear<'a, T, I, O> 
    where 
        D: Alloc<T> + GraphReturn + 'a
    {
        let config = args.into_config();
        let (weights, bias) = config.init_params(device);

        Linear {
            weights,
            bias,
            dweights: None,
            dbias: None,
            inputs: None,
            l2_reg: config.l2_reg,
            l2_reg_loss: config.l2_reg_loss
        }
    }

    pub fn set_l2_reg_loss(&mut self, l2_reg_loss: &'a RefCell<T>) -> &mut Self {
        self.l2_reg_loss = Some(l2_reg_loss);
        self
    }
}

impl<'a, T: Copy + Float, const I: usize, const O: usize> WithDevice<'a, T>
    for Linear<'a, T, I, O>
{
    fn with<'b: 'a, D: Alloc<T> + GraphReturn>(device: &'b D) -> Self
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
        
        if let Some(bias) = &self.bias {
            forward.add_row_mut(bias);
        }

        // l2 reg loss
        if let Some(l2_reg_loss) = self.l2_reg_loss {
            let mut l2_reg_loss = l2_reg_loss.borrow_mut();
            
            *l2_reg_loss += (&self.weights * &self.weights).sum() * self.l2_reg;

            
            if let Some(bias) = &self.bias {
                *l2_reg_loss += (bias * bias).sum() * self.l2_reg;
            }
        }

        forward
    }

    pub fn backward(&mut self, grad: &Matrix<'a, T>) -> Matrix<'a, T>
    where
        T: CudaTranspose,
    {
        if self.bias.is_some() {
            self.dbias = Some(grad.sum_rows());
        }

        let mut dweights = self.inputs.as_ref().unwrap().T().gemm(grad);
        
        if self.l2_reg > T::zero() {
            dweights += &self.weights * (self.l2_reg * T::two());

            if let Some(dbias) = &mut self.dbias {
                *dbias += self.bias.as_ref().unwrap() * (self.l2_reg * T::two())
            }
        }
        self.dweights = Some(dweights);

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
            l2_reg_loss: Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::linear::{Linear, LinearConfig, Glorot};
    use custos::CPU;

    #[test]
    fn test_bias() {
        let device = CPU::new();

        let linear = Linear::<f32, 8, 16>::new(&device, LinearConfig {
            init: Glorot::new(),
            bias: false,
            ..Default::default()
        });

        assert!(linear.bias.is_none());
    }
}
