use custos::{number::Float, Alloc, GraphReturn};
use custos_math::Matrix;

pub(super) trait Init<'a, T, D> {
    fn init(&self, device: &'a D) -> (Matrix<'a, T>, Matrix<'a, T>);
}

pub struct RandomUniform<T, const I: usize, const O: usize> {
    pub min: T,
    pub max: T,
}

impl<'a, T, D, const I: usize, const O: usize> Init<'a, T, D> for RandomUniform<T, I, O> 
where 
    T: Float,
    D: Alloc<T> + GraphReturn 
{
    fn init(&self, device: &'a D) -> (Matrix<'a, T>, Matrix<'a, T>) {
        let mut weights = Matrix::<T>::from((device, I, O));
        weights.rand(self.min, self.max);

        let bias = Matrix::<T>::from((device, 1, O));
        (weights, bias)
    }
}

pub struct Glorot<const I: usize, const O: usize> {}

impl<'a, T: Float, D: Alloc<T> + GraphReturn, const I: usize, const O: usize> Init<'a, T, D> for Glorot<I, O> {
    fn init(&self, device: &'a D) -> (Matrix<'a, T>, Matrix<'a, T>) {
        let mut weights = Matrix::<T>::from((device, I, O));

        let glorot = (T::from_usize(6) / T::from_usize(I + O)).sqrt();
        
        weights.rand(glorot.negate(), glorot);
        let bias = Matrix::<T>::from((device, 1, O));

        (weights, bias)

    }
}
