use custos::{GraphReturn, number::Float, Alloc};
use custos_math::Matrix;
use crate::Glorot;
use super::init::Init;


pub struct LinearConfig<'a, T, D, const I: usize, const O: usize> {
    init: Box<dyn Init<'a, T, D>>,
    pub bias: bool,
    pub l2_reg: T
}

impl<'a, T, D, const I: usize, const O: usize> LinearConfig<'a, T, D, I, O> {
    pub fn init_params(&self, device: &'a D) -> (Matrix<'a, T>, Matrix<'a, T>) {
        self.init.init(device)
    }
}


impl<'a, T: Float, D: Alloc<T> + GraphReturn, const I: usize, const O: usize> Default for LinearConfig<'a, T, D, I, O> {
    fn default() -> Self {
        let glorot: Glorot<I, O> = Glorot {};
        Self {
            init: Box::new(glorot),
            bias: true,
            l2_reg: T::default()
        }
    }
}

pub trait AsLinearConfig<'a, T, D, const I: usize, const O: usize> {
    fn as_linear_config(self) -> LinearConfig<'a, T, D, I, O>;
}

impl<'a, T, D, const I: usize, const O: usize> AsLinearConfig<'a, T, D, I, O> for () 
where 
    T: Float, 
    D: Alloc<T> + GraphReturn 
{
    fn as_linear_config(self) -> LinearConfig<'a, T, D, I, O> {
        LinearConfig::default()
    }
}

impl<'a, T: Copy, D, const I: usize, const O: usize> AsLinearConfig<'a, T, D, I, O> 
    for LinearConfig<'a, T, D, I, O> 
    
    {
    fn as_linear_config(self) -> Self {
        LinearConfig { init: self.init, bias: self.bias, l2_reg: self.l2_reg }
    }
}