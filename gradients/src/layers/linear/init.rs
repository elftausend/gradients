use custos_math::custos::{number::Float, Alloc, Device, Dim2, WithShape};
use custos_math::{Matrix, RandOp, custos};

use super::{LinearParams, LinearParams2};

pub trait Init<'a, T, D: Device, const I: usize, const O: usize> {
    fn init(&self, device: &'a D, with_bias: bool) -> LinearParams<'a, T, D>;
}

pub trait Init2<'a, T, D: Device, const I: usize, const O: usize> {
    fn init2(&self, device: &'a D, with_bias: bool) -> LinearParams2<'a, T, D, I, O>;
}

#[derive(Debug)]
pub struct RandomUniform<T> {
    pub min: T,
    pub max: T,
}

impl<T> RandomUniform<T> {
    pub fn new(min: T, max: T) -> RandomUniform<T> {
        RandomUniform { min, max }
    }
    pub fn one() -> RandomUniform<T>
    where
        T: Float,
    {
        RandomUniform {
            min: -T::one(),
            max: T::one(),
        }
    }
}

impl<'a, T, D, const I: usize, const O: usize> Init2<'a, T, D, I, O> for RandomUniform<T>
where
    T: Copy,
    D: Alloc<'a, T, Dim2<I, O>> + RandOp<T, Dim2<I, O>> + 'a,
    custos::Buffer<'a, T, D, Dim2<I, O>>: WithShape<&'a D, ()>,
    custos::Buffer<'a, T, D, Dim2<1, O>>: WithShape<&'a D, ()>,
{
    fn init2(&self, device: &'a D, with_bias: bool) -> LinearParams2<'a, T, D, I, O> {
        let mut weights = Matrix::with(device, ());
        weights.rand(self.min, self.max);

        let mut bias = None;
        if with_bias {
            bias = Some(Matrix::with(device, ()));
        }

        (weights, bias)
    }
}

impl<'a, T, D, const I: usize, const O: usize> Init<'a, T, D, I, O> for RandomUniform<T>
where
    T: Float,
    D: Alloc<'a, T> + RandOp<T>,
{
    fn init(&self, device: &'a D, with_bias: bool) -> LinearParams<'a, T, D> {
        let mut weights = Matrix::from((device, I, O));
        weights.rand(self.min, self.max);

        let mut bias = None;
        if with_bias {
            bias = Some(Matrix::from((device, 1, O)));
        }

        (weights, bias)
    }
}

pub struct Glorot;

impl Glorot {
    #[inline]
    pub fn new() -> Self {
        Glorot
    }
}

impl<'a, T, D, const I: usize, const O: usize> Init2<'a, T, D, I, O> for Glorot
where
    T: Float,
    D: Alloc<'a, T, Dim2<I, O>> + RandOp<T, Dim2<I, O>> + 'a,
    custos::Buffer<'a, T, D, Dim2<I, O>>: WithShape<&'a D, ()>,
    custos::Buffer<'a, T, D, Dim2<1, O>>: WithShape<&'a D, ()>,
{
    fn init2(&self, device: &'a D, with_bias: bool) -> LinearParams2<'a, T, D, I, O> {
        let mut weights = Matrix::with(device, ());

        let glorot = (T::from_usize(6) / T::from_usize(I + O)).sqrt();

        weights.rand(-glorot, glorot);

        let mut bias = None;
        if with_bias {
            bias = Some(Matrix::with(device, ()));
        }

        (weights, bias)
    }
}

impl<'a, T, D, const I: usize, const O: usize> Init<'a, T, D, I, O> for Glorot
where
    T: Float,
    D: Alloc<'a, T> + RandOp<T>,
{
    fn init(&self, device: &'a D, with_bias: bool) -> LinearParams<'a, T, D> {
        let mut weights = Matrix::<T, D>::from((device, I, O));

        let glorot = (T::from_usize(6) / T::from_usize(I + O)).sqrt();

        weights.rand(-glorot, glorot);

        let mut bias = None;
        if with_bias {
            bias = Some(Matrix::<T, D>::from((device, 1, O)));
        }

        (weights, bias)
    }
}
