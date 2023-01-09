use crate::{GetParam, WithDevice};
use custos::{CloneBuf, Device, ShallowCopy, Shape};
use custos_math::{
    nn::{ActivationOps, SoftmaxOps},
    AssignOps, BaseOps, Matrix,
};
use gradients_derive::NoParams;

pub trait Activation<'a, T, D: Device, S: Shape = ()> {
    fn forward(inputs: Matrix<'a, T, D, S>) -> Matrix<'a, T, D, S>;
    fn backward(
        inputs: &mut Matrix<'a, T, D, S>,
        grads: Matrix<'a, T, D, S>,
    ) -> Matrix<'a, T, D, S>;
}

impl<'a, T, D: Device + 'a, S: Shape> Activation<'a, T, D, S> for () {
    #[inline]
    fn forward(inputs: Matrix<'a, T, D, S>) -> Matrix<'a, T, D, S> {
        inputs
    }

    #[inline]
    fn backward(
        _inputs: &mut Matrix<'a, T, D, S>,
        grads: Matrix<'a, T, D, S>,
    ) -> Matrix<'a, T, D, S> {
        grads
    }
}

pub struct ReLU2;

impl<'a, T, D: ActivationOps<T, S> + AssignOps<T, S>, S: Shape> Activation<'a, T, D, S> for ReLU2 {
    #[inline]
    fn forward(mut inputs: Matrix<'a, T, D, S>) -> Matrix<'a, T, D, S> {
        inputs.relu_mut();
        inputs
    }

    fn backward(
        inputs: &mut Matrix<'a, T, D, S>,
        mut grads: Matrix<'a, T, D, S>,
    ) -> Matrix<'a, T, D, S> {
        inputs.relu_grad_mut();
        grads *= &*inputs;
        grads
    }
}

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

impl<'a, T: Clone, D: CloneBuf<'a, T> + SoftmaxOps<T>> Softmax<'a, T, D> {
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
