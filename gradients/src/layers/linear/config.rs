use std::cell::RefCell;

use super::{init::Init, LinearParams};
use crate::linear::Glorot;
use custos::{number::Float, Alloc, Device, GraphReturn};
use custos_math::RandOp;

pub struct LinearConfig<'a, T, D, const I: usize, const O: usize> {
    pub init: Box<dyn Init<'a, T, D, I, O>>,
    pub bias: bool,
    pub l2_reg: T,
    pub l2_reg_loss: Option<&'a RefCell<T>>,
}

impl<'a, T, D: Device, const I: usize, const O: usize> LinearConfig<'a, T, D, I, O> {
    pub fn init_params(&self, device: &'a D) -> LinearParams<'a, T, D> {
        self.init.init(device, self.bias)
    }
}

impl<'a, T: Float, D: Alloc<'a, T> + RandOp<T>, const I: usize, const O: usize>
    Default for LinearConfig<'a, T, D, I, O>
{
    fn default() -> Self {
        Self {
            init: Box::new(Glorot),
            bias: true,
            l2_reg: T::default(),
            l2_reg_loss: None,
        }
    }
}

pub trait IntoLinearConfig<'a, T, D: 'a, const I: usize, const O: usize> {
    fn into_config(self) -> LinearConfig<'a, T, D, I, O>;
}

impl<'a, T, D, const I: usize, const O: usize> IntoLinearConfig<'a, T, D, I, O> for ()
where
    T: Float,
    D: Alloc<'a, T> + 'a + RandOp<T>,
{
    fn into_config(self) -> LinearConfig<'a, T, D, I, O> {
        LinearConfig::default()
    }
}

impl<'a, T, D: 'a, const I: usize, const O: usize> IntoLinearConfig<'a, T, D, I, O>
    for LinearConfig<'a, T, D, I, O>
{
    fn into_config(self) -> LinearConfig<'a, T, D, I, O> {
        self
    }
}

pub struct Bias(pub bool);

impl<'a, T, D, const I: usize, const O: usize> IntoLinearConfig<'a, T, D, I, O> for Bias
where
    T: Float,
    D: Alloc<'a, T> + GraphReturn + 'a + RandOp<T>,
{
    fn into_config(self) -> LinearConfig<'a, T, D, I, O> {
        LinearConfig {
            bias: self.0,
            ..Default::default()
        }
    }
}

pub struct L2<T>(pub T);

impl<'a, T, D, const I: usize, const O: usize> IntoLinearConfig<'a, T, D, I, O> for L2<T>
where
    T: Float,
    D: Alloc<'a, T> + GraphReturn + 'a + RandOp<T>,
{
    fn into_config(self) -> LinearConfig<'a, T, D, I, O> {
        LinearConfig {
            l2_reg: self.0,
            ..Default::default()
        }
    }
}
