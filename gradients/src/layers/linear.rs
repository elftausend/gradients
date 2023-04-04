mod config;
mod init;
mod l2_reg;

use std::{
    cell::RefCell,
    marker::PhantomData,
    ops::{AddAssign, Mul},
};

pub use config::*;
pub use init::{Glorot, RandomUniform};
pub use l2_reg::*;

use custos_math::custos::{
    number::Float, prelude::Number, Alloc, CloneBuf, Device, Dim2,
    IsShapeIndep, MayDim2, PtrConv, ShallowCopy, Shape, ToDim, CPU, CommonPtrs, self,
};
use custos_math::{
    AdditionalOps, AssignOps, BaseOps, Gemm, Matrix, RandOp, RowOp, SumOps,
    SumOverOps, TransposeOp,
};

use crate::{Activation, GetParam, Param, Params, WithDevice};

type LinearParams<'a, T, D> = (Matrix<'a, T, D>, Option<Matrix<'a, T, D>>);

type LinearParams2<'a, T, D, const I: usize, const O: usize> = (
    Matrix<'a, T, D, Dim2<I, O>>,
    Option<Matrix<'a, T, D, Dim2<1, O>>>,
);

#[rustfmt::skip]
pub struct Linear2<'a, T, const I: usize, const O: usize, D = CPU, A = (), const SAMPLES: usize=1> 
where 
    D: Device, 
    //A: Activation<'a, T, D, Dim2<SAMPLES, O>>
{
    pub weights: Matrix<'a, T, D, Dim2<I, O>>,
    bias: Option<Matrix<'a, T, D, Dim2<1, O>>>,
    inputs: Option<Matrix<'a, T, D, Dim2<SAMPLES, I>>>,
    activation_inputs: Option<Matrix<'a, T, D, Dim2<SAMPLES, O>>>,
    dweights: Option<Matrix<'a, T, D>>,
    dbias: Option<Matrix<'a, T, D>>,
    pub l2_reg: T,
    _p: PhantomData<A>
}

impl<'a, T, const I: usize, const O: usize, D, A, const SAMPLES: usize>
    Linear2<'a, T, I, O, D, A, SAMPLES>
where
    T: Number,
    D: Device,
    // A: Activation<'a, T, D, Dim2<SAMPLES, O>>,
{
    pub fn new(device: &'a D, config: LinearConfig2<'a, T, D, I, O>) -> Self
    where
        D: Alloc<'a, T, Dim2<I, O>>,
    {
        let (weights, bias) = config.init_params2(device);
        Linear2 {
            weights,
            bias,
            inputs: None,
            activation_inputs: None,
            dweights: None,
            dbias: None,
            l2_reg: T::default(),
            _p: PhantomData,
        }
    }

    pub fn forward<IS>(
        &mut self,
        inputs: &Matrix<'a, T, D, IS>,
    ) -> Matrix<'a, T, D, Dim2<SAMPLES, O>>
    where
        D: CloneBuf<'a, T, IS>
            + CloneBuf<'a, T, Dim2<SAMPLES, O>>
            + Gemm<T, IS, Dim2<I, O>, Dim2<SAMPLES, O>>
            + RowOp<T, Dim2<SAMPLES, O>, Dim2<1, O>, D>
            + ToDim<T, IS, Dim2<SAMPLES, I>>,
        D::Ptr<T, IS>: ShallowCopy,
        D::Ptr<T, Dim2<SAMPLES, O>>: ShallowCopy,
        IS: MayDim2<SAMPLES, I>,
        A: Activation<'a, T, D, Dim2<SAMPLES, O>>,
        // debug: <D as custos::Device>::Ptr<T, Dim2<SAMPLES, I>>: CommonPtrs<T>
    {
        // this shallow clone is not always fine (create inputs that live shorter than Linear):
        // (shallow clone of cached vars is ok)
        self.inputs = Some(inputs.shallow_or_clone().to_dims());
        //println!("Use inputs ptr: {:?}", self.inputs.as_ref().unwrap().data.ptrs().0);
        
        //self.inputs = Some(inputs.clone().to_dims());

        let mut forward = inputs.gemm(&self.weights);

        if let Some(bias) = &self.bias {
            forward.add_row_mut(bias);
        }

        //self.activation_inputs = Some(forward.shallow_or_clone());
        self.activation_inputs = Some(forward.clone());

        // activation function
        A::forward(forward)
    }

    pub fn backward<IS>(&mut self, grads: Matrix<'a, T, D, IS>) -> Matrix<'a, T, D>
    where
        D: TransposeOp<T, Dim2<I, O>>
            + PtrConv // TODO: IsShapeIndep
            + TransposeOp<T, Dim2<SAMPLES, I>>
            + ToDim<T, IS, Dim2<SAMPLES, O>>
            + AdditionalOps<T>
            + AssignOps<T>
            + Gemm<T>
            + RowOp<T>
            + SumOverOps<T>,
        A: Activation<'a, T, D>,
        IS: Shape,
        D::Ptr<T, IS>: ShallowCopy
    {
        let grads = grads.to_dims::<()>();

        // activation fn backwards
        let grads = A::backward(
            self.activation_inputs.as_mut().unwrap().as_dims_mut(),
            grads,
        );

        if self.bias.is_some() {
            self.dbias = Some(grads.sum_rows());
        }

        let mut dweights = self.inputs.as_ref().unwrap().T().gemm(&grads);

        if self.l2_reg > T::zero() {
            dweights += self.weights.as_dims() * (self.l2_reg * T::two());

            if let Some(dbias) = &mut self.dbias {
                *dbias += self.bias.as_ref().unwrap().as_dims() * (self.l2_reg * T::two())
            }
        }
        self.dweights = Some(dweights);

        grads.gemm(&self.weights.T())
    }

    pub fn params<'b>(&'b mut self) -> Params<'b, T, D>
    where
        D: IsShapeIndep,
    {
        let dweights = self
            .dweights
            .as_ref()
            .expect(".backward() on this layer should be called at this moment");

        let weights = self.weights.as_dims_mut();

        let bias = self.bias.as_mut().map(|bias| bias.as_dims_mut());

        Params {
            weights,
            bias,
            dweights,
            dbias: self.dbias.as_ref(),
        }
    }
}

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
    use custos_math::custos::CPU;

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
