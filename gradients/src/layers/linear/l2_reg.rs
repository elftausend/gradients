use std::{cell::RefCell, fmt::Display};

use custos::{number::Float, Alloc, GraphReturn};

use crate::linear::{IntoLinearConfig, LinearConfig};

#[derive(Debug, Clone)]
pub struct L2Reg<T> {
    pub l2: T,
    pub loss: RefCell<T>,
}

impl<T: Default> L2Reg<T> {
    pub fn new(l2: T) -> Self {
        L2Reg {
            l2,
            loss: RefCell::new(T::default()),
        }
    }
    pub fn zero(&self) {
        *self.loss.borrow_mut() = T::default();
    }
}

impl<T: Display> Display for L2Reg<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.loss.borrow())
    }
}

impl<'a, T, D, const I: usize, const O: usize> IntoLinearConfig<'a, T, D, I, O> for &'a L2Reg<T>
where
    T: Float,
    D: Alloc<T> + GraphReturn + 'a,
{
    fn into_config(self) -> LinearConfig<'a, T, D, I, O> {
        LinearConfig {
            l2_reg: self.l2,
            l2_reg_loss: Some(&self.loss),
            ..Default::default()
        }
    }
}

pub struct L2Loss<'a, T> {
    pub loss: &'a RefCell<T>,
}

impl<'a, T, D, const I: usize, const O: usize> IntoLinearConfig<'a, T, D, I, O> for L2Loss<'a, T>
where
    T: Float,
    D: Alloc<T> + GraphReturn + 'a,
{
    fn into_config(self) -> LinearConfig<'a, T, D, I, O> {
        LinearConfig {
            l2_reg_loss: Some(self.loss),
            ..Default::default()
        }
    }
}
