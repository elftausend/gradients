use core::cell::RefCell;

use super::{
    init::{Init, Init2},
    LinearParams, LinearParams2,
};
use crate::linear::Glorot;
use custos_math::custos::{number::Float, Alloc, Device, Dim2, WithShape, self};
use custos_math::RandOp;

pub struct LinearConfig2<'a, T, D, const I: usize, const O: usize> {
    pub init: &'a dyn Init2<'a, T, D, I, O>,
    pub bias: bool,
    pub l2_reg: T,
    pub l2_reg_loss: Option<&'a RefCell<T>>,
}

impl<'a, T, D: Device, const I: usize, const O: usize> LinearConfig2<'a, T, D, I, O> {
    pub fn init_params2(&self, device: &'a D) -> LinearParams2<'a, T, D, I, O> {
        self.init.init2(device, self.bias)
    }
}

impl<'a, T, D, const I: usize, const O: usize> Default for LinearConfig2<'a, T, D, I, O>
where
    T: Float,
    D: Alloc<'a, T, Dim2<I, O>> + RandOp<T, Dim2<I, O>> + 'a,
    custos::Buffer<'a, T, D, Dim2<I, O>>: WithShape<&'a D, ()>,
    custos::Buffer<'a, T, D, Dim2<1, O>>: WithShape<&'a D, ()>,
{
    fn default() -> Self {
        Self {
            init: &Glorot, // Glorot
            bias: true, // false
            l2_reg: T::default(),
            l2_reg_loss: None,
        }
    }
}

pub struct LinearConfig<'a, T, D, const I: usize, const O: usize> {
    pub init: &'a dyn Init<'a, T, D, I, O>,
    pub bias: bool,
    pub l2_reg: T,
    pub l2_reg_loss: Option<&'a RefCell<T>>,
}

impl<'a, T, D: Device, const I: usize, const O: usize> LinearConfig<'a, T, D, I, O> {
    pub fn init_params(&self, device: &'a D) -> LinearParams<'a, T, D> {
        self.init.init(device, self.bias)
    }
}

impl<'a, T: Float, D: Alloc<'a, T> + RandOp<T>, const I: usize, const O: usize> Default
    for LinearConfig<'a, T, D, I, O>
{
    fn default() -> Self {
        Self {
            init: &Glorot,
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
    D: Alloc<'a, T> + 'a + RandOp<T>,
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
    D: Alloc<'a, T> + 'a + RandOp<T>,
{
    fn into_config(self) -> LinearConfig<'a, T, D, I, O> {
        LinearConfig {
            l2_reg: self.0,
            ..Default::default()
        }
    }
}

macro_rules! impl_into_linear_conf {
    ($($impl_type:ty),*) => {
        $(
            impl<'a, T: Float, D: Alloc<'a, T>  + 'a + RandOp<T>, const I: usize, const O: usize> IntoLinearConfig<'a, T, D, I, O> for &'a $impl_type {
                fn into_config(self) -> LinearConfig<'a, T, D, I, O> {
                    LinearConfig {
                        init: self,
                        ..Default::default()
                    }
                }
            }
        )*
    };
}

impl_into_linear_conf! {super::RandomUniform<T>, Glorot}
