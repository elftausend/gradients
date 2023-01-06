mod config;
mod init;
mod l2_reg;

use std::{
    cell::RefCell,
    ops::{AddAssign, Mul},
};

pub use config::*;
pub use init::{Glorot, RandomUniform};
pub use l2_reg::*;

use custos::{
    number::Float, prelude::Number, Alloc, CDatatype, CloneBuf, Device, GenericBlas, ShallowCopy,
};
use custos_math::{
    AdditionalOps, AssignOps, BaseOps, CudaTranspose, Gemm, Matrix, RandOp, RowOp, SumOps,
    SumOverOps, TransposeOp,
};

use crate::{GetParam, Param, WithDevice};

type LinearParams<'a, T, D> = (Matrix<'a, T, D>, Option<Matrix<'a, T, D>>);

// TODO: remove default types
pub struct Linear<
    'a,
    T,
    const I: usize,
    const O: usize,
    D: Device = custos::CPU,
    const SAMPLES: usize = 1,
> {
    pub weights: Matrix<'a, T, D>,
    pub bias: Option<Matrix<'a, T, D>>,
    pub dweights: Option<Matrix<'a, T, D>>,
    pub dbias: Option<Matrix<'a, T, D>>,
    inputs: Option<Matrix<'a, T, D>>,
    pub l2_reg: T,
    pub l2_reg_loss: Option<&'a RefCell<T>>,
}

impl<'a, T: Copy + Float, D, const I: usize, const O: usize, const SAMPLES: usize>
    Linear<'a, T, I, O, D, SAMPLES>
where
    D: Device + 'a,
{
    pub fn new<'b: 'a>(
        device: &'b D,
        args: impl IntoLinearConfig<'a, T, D, I, O>,
    ) -> Linear<'a, T, I, O, D, SAMPLES> {
        let config = args.into_config();
        let (weights, bias) = config.init_params(device);

        Linear {
            weights,
            bias,
            dweights: None,
            dbias: None,
            inputs: None,
            l2_reg: config.l2_reg,
            l2_reg_loss: config.l2_reg_loss,
        }
    }

    pub fn set_l2_reg_loss(&mut self, l2_reg_loss: &'a RefCell<T>) -> &mut Self {
        self.l2_reg_loss = Some(l2_reg_loss);
        self
    }
}

impl<'a, T, D, const I: usize, const O: usize, const SAMPLES: usize> WithDevice<'a, T, D>
    for Linear<'a, T, I, O, D, SAMPLES>
where
    T: Copy + Float,
    D: Alloc<'a, T> + RandOp<T>,
{
    fn with<'b: 'a>(device: &'b D) -> Self
    where
        Self: Default,
    {
        Self::new(device, ())
    }
}

impl<'a, T, D: Device, const I: usize, const O: usize, const SAMPLES: usize>
    Linear<'a, T, I, O, D, SAMPLES>
where
    T: AddAssign + Mul<Output = T> + Copy,
{
    pub fn forward(&mut self, inputs: &Matrix<'a, T, D>) -> Matrix<'a, T, D>
    where
        D: CloneBuf<'a, T> + Gemm<T> + RowOp<T> + BaseOps<T> + SumOps<T>,
        D::Ptr<T, ()>: ShallowCopy,
    {
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

    pub fn backward(&mut self, grad: &Matrix<'a, T, D>) -> Matrix<'a, T, D>
    where
        T: Number,
        D: TransposeOp<T> + AdditionalOps<T> + AssignOps<T> + Gemm<T> + RowOp<T> + SumOverOps<T>,
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

impl<'a, T: Copy, D: Device, const I: usize, const O: usize> GetParam<'a, T, D>
    for Linear<'a, T, I, O, D>
where
    D::Ptr<T, ()>: ShallowCopy,
{
    fn params(&mut self) -> Option<Param<'a, T, D>> {
        Some(Param::new(
            self.weights.shallow(),
            self.bias.as_ref().map(|bias| bias.shallow()),
            self.dweights.as_ref().unwrap().shallow(),
            self.dbias.as_ref().unwrap().shallow(),
        ))
    }
}

impl<'a, T, D: Device, const I: usize, const O: usize> Default for Linear<'a, T, I, O, D> {
    fn default() -> Self {
        unimplemented!();
    }
}

#[cfg(test)]
mod tests {
    use crate::linear::{Glorot, Linear, LinearConfig};
    use custos::CPU;

    #[test]
    fn test_bias() {
        let device = CPU::new();

        let linear = Linear::<f32, 8, 16>::new(
            &device,
            LinearConfig {
                init: &Glorot,
                bias: false,
                ..Default::default()
            },
        );

        assert!(linear.bias.is_none());
    }
}
