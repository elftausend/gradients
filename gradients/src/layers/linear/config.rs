use super::{init::Init, LinearParams};
use crate::Glorot;
use custos::{number::Float, Alloc, GraphReturn};

pub struct LinearConfig<'a, T, D, const I: usize, const O: usize> {
    pub init: Box<dyn Init<'a, T, D, I, O>>,
    pub bias: bool,
    pub l2_reg: T,
}

impl<'a, T, D, const I: usize, const O: usize> LinearConfig<'a, T, D, I, O> {
    pub fn init_params(&self, device: &'a D) -> LinearParams<'a, T> {
        self.init.init(device, self.bias)
    }
}

impl<'a, T: Float, D: Alloc<T> + GraphReturn, const I: usize, const O: usize> Default
    for LinearConfig<'a, T, D, I, O>
{
    fn default() -> Self {
        Self {
            init: Box::new(Glorot),
            bias: true,
            l2_reg: T::default(),
        }
    }
}

pub struct LinearArg<'a, T, D, const I: usize, const O: usize> {
    pub device: &'a D,
    pub config: LinearConfig<'a, T, D, I, O>,
}

pub trait LinearArgs<'a, T, D: 'a, const I: usize, const O: usize> {
    fn linear_arg(self) -> LinearArg<'a, T, D, I, O>;
}

impl<'a, T, D, const I: usize, const O: usize> LinearArgs<'a, T, D, I, O> for &'a D 
where 
    T: Float, 
    D: Alloc<T> + GraphReturn 
{
    fn linear_arg(self) -> LinearArg<'a, T, D, I, O> {
        LinearArg {
            device: self,
            config: LinearConfig::default(),
        }
    }
}

impl<'a, T, D, const I: usize, const O: usize> LinearArgs<'a, T, D, I, O> for (&'a D, LinearConfig<'a, T, D, I, O> )
where 
    T: Float, 
    D: Alloc<T> + GraphReturn 
{
    fn linear_arg(self) -> LinearArg<'a, T, D, I, O> {
        LinearArg {
            device: self.0,
            config: self.1,
        }
    }
}