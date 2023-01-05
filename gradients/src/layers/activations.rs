use crate::{GetParam, WithDevice};
use custos::{CloneBuf, Device, ShallowCopy};
use custos_math::{
    nn::{ActivationOps, SoftmaxOps},
    BaseOps, Matrix,
};
use gradients_derive::NoParams;

#[derive(NoParams)]
pub struct ReLU<'a, T, D: Device> {
    inputs: Option<Matrix<'a, T, D>>,
}

impl<'a, T: Clone, D: CloneBuf<'a, T> + ActivationOps<T>> ReLU<'a, T, D> {
    pub fn new() -> ReLU<'a, T, D> {
        ReLU { inputs: None }
    }
    pub fn forward(&mut self, inputs: &Matrix<'a, T, D>) -> Matrix<'a, T, D>
    where
        D::Ptr<T, ()>: ShallowCopy,
    {
        self.inputs = Some(inputs.shallow_or_clone());
        inputs.relu()
    }
    pub fn backward(&self, grad: &Matrix<'a, T, D>) -> Matrix<'a, T, D>
    where
        D: BaseOps<T>,
    {
        self.inputs.as_ref().unwrap().relu_grad() * grad
    }
}

impl<'a, T, D: Device> Default for ReLU<'a, T, D> {
    fn default() -> Self {
        Self {
            inputs: Default::default(),
        }
    }
}

#[derive(NoParams)]
pub struct Softmax<'a, T, D: Device> {
    activated: Option<Matrix<'a, T, D>>,
}

impl<'a, T: Clone, D: CloneBuf<'a, T> + SoftmaxOps<T>>
    Softmax<'a, T, D>
{
    pub fn new() -> Self {
        Softmax { activated: None }
    }

    pub fn forward(&mut self, x: &Matrix<'a, T, D>) -> Matrix<'a, T, D>
    where
        D::Ptr<T, ()>: ShallowCopy,
    {
        let activated = x.softmax();
        self.activated = Some(activated.shallow_or_clone());
        activated
    }

    pub fn backward(&self, grad: &Matrix<'a, T, D>) -> Matrix<'a, T, D>
    where
        D: BaseOps<T>,
    {
        grad.softmax_grad(self.activated.as_ref().unwrap())
    }
}

impl<'a, T, D: Device> Default for Softmax<'a, T, D> {
    fn default() -> Self {
        Self {
            activated: Default::default(),
        }
    }
}

#[derive(NoParams)]
pub struct Tanh<'a, T, D: Device> {
    inputs: Option<Matrix<'a, T, D>>,
}

impl<'a, T, D: Device> Tanh<'a, T, D> {
    pub fn new() -> Tanh<'a, T, D> {
        Tanh { inputs: None }
    }
}

impl<'a, T: Clone, D: CloneBuf<'a, T> + ActivationOps<T>> Tanh<'a, T, D> {
    pub fn forward(&mut self, inputs: &Matrix<'a, T, D>) -> Matrix<'a, T, D>
    where
        D::Ptr<T, ()>: ShallowCopy,
    {
        self.inputs = Some(inputs.shallow_or_clone());
        inputs.tanh()
    }
    pub fn backward(&self, grad: &Matrix<'a, T, D>) -> Matrix<'a, T, D>
    where
        D: BaseOps<T>,
    {
        self.inputs.as_ref().unwrap().tanh_grad() * grad
    }
}

impl<'a, T, D: Device> Default for Tanh<'a, T, D> {
    fn default() -> Self {
        Self {
            inputs: Default::default(),
        }
    }
}

#[derive(NoParams)]
pub struct Sigmoid<'a, T, D: Device> {
    activated: Option<Matrix<'a, T, D>>,
}

impl<'a, T, D: Device> Sigmoid<'a, T, D> {
    pub fn new() -> Sigmoid<'a, T, D> {
        Sigmoid::default()
    }
}

impl<'a, T: Clone, D: CloneBuf<'a, T> + ActivationOps<T>> Sigmoid<'a, T, D> {
    pub fn forward(&mut self, inputs: &Matrix<'a, T, D>) -> Matrix<'a, T, D>
    where
        D::Ptr<T, ()>: ShallowCopy,
    {
        let activated = inputs.sigmoid();
        self.activated = Some(inputs.shallow_or_clone());
        activated
    }

    pub fn backward(&mut self, grad: &Matrix<'a, T, D>) -> Matrix<'a, T, D>
    where
        D: BaseOps<T>,
    {
        self.activated.as_ref().unwrap().sigmoid_grad() * grad
    }
}

impl<'a, T, D: Device> Default for Sigmoid<'a, T, D> {
    #[inline]
    fn default() -> Self {
        Self {
            activated: Default::default(),
        }
    }
}
