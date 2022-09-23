use custos::{number::Float, Alloc, GraphReturn};
use custos_math::Matrix;

use super::LinearParams;

pub trait Init<'a, T, D, const I: usize, const O: usize> {
    fn init(&self, device: &'a D, with_bias: bool) -> LinearParams<'a, T>;
}

pub struct RandomUniform<T> {
    pub min: T,
    pub max: T,
}

impl<T> RandomUniform<T> {
    pub fn new(min: T, max: T) -> Box<RandomUniform<T>> {
        Box::new(
            RandomUniform { 
                min, 
                max 
            }
        )
    }
    pub fn one() -> Box<RandomUniform<T>> 
    where T: Float
    {
        Box::new(
            RandomUniform { 
                min: -T::one(), 
                max: T::one() 
            }
        )
    }
}

impl<'a, T, D, const I: usize, const O: usize> Init<'a, T, D, I, O> for RandomUniform<T>
where
    T: Float,
    D: Alloc<T> + GraphReturn,
{
    fn init(&self, device: &'a D, with_bias: bool) -> LinearParams<'a, T> {
        let mut weights = Matrix::<T>::from((device, I, O));
        weights.rand(self.min, self.max);

        let mut bias = None;
        if with_bias {
            bias = Some(Matrix::<T>::from((device, 1, O)));
        }

        (weights, bias)
    }
}

pub struct Glorot;

impl Glorot {
    pub fn new() -> Box<Self> {
        Box::new(Self {})
    }
}

impl<'a, T: Float, D: Alloc<T> + GraphReturn, const I: usize, const O: usize> Init<'a, T, D, I, O>
    for Glorot
{
    fn init(&self, device: &'a D, with_bias: bool) -> LinearParams<'a, T> {
        let mut weights = Matrix::<T>::from((device, I, O));

        let glorot = (T::from_usize(6) / T::from_usize(I + O)).sqrt();

        weights.rand(-glorot, glorot);
        
        let mut bias = None;
        if with_bias {
            bias = Some(Matrix::<T>::from((device, 1, O)));
        }
        
        (weights, bias)
    }
}
