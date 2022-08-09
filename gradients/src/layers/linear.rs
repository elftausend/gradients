use custos::{number::Float, Alloc, CDatatype, GenericBlas};
use custos_math::{CudaTranspose, Matrix};

use crate::{GetParam, Param, WithDevice};

pub struct Linear<'a, T, const I: usize, const O: usize> {
    pub weights: Matrix<'a, T>,
    pub bias: Matrix<'a, T>,
    pub dweights: Option<Matrix<'a, T>>,
    pub dbias: Option<Matrix<'a, T>>,
    inputs: Option<Matrix<'a, T>>,
}

impl<'a, T: Copy + Float, const I: usize, const O: usize> Linear<'a, T, I, O> {
    pub fn new<D: Alloc<T>>(device: &'a D) -> Linear<'a, T, I, O> {
        let mut weights = Matrix::<T>::from((device, I, O));

        let glorot = (T::from_usize(6) / T::from_usize(I + O)).sqrt();
        weights.rand(glorot.negate(), glorot);

        //let weights = weights.muls(weight_size);
        //let weights = weights + (T::one() / T::from_usize(100));

        let bias = Matrix::<T>::from((device, 1, O));

        Linear {
            weights,
            bias,
            dweights: None,
            dbias: None,
            inputs: None,
        }
    }
}

impl<'a, T: Copy + Float, const I: usize, const O: usize> WithDevice<'a, T>
    for Linear<'a, T, I, O>
{
    fn with_device<D: Alloc<T>>(device: &'a D) -> Self
    where
        Self: Default,
    {
        Self::new(device)
    }
}

impl<'a, T: Float + GenericBlas + CDatatype, const I: usize, const O: usize> Linear<'a, T, I, O> {
    pub fn forward(&mut self, inputs: &Matrix<'a, T>) -> Matrix<'a, T> {
        self.inputs = Some(inputs.shallow());
        let mut forward = inputs.gemm(&self.weights);
        forward.add_row_mut(&self.bias);
        forward
    }

    pub fn backward(&mut self, grad: &Matrix<'a, T>) -> Matrix<'a, T>
    where
        T: CudaTranspose,
    {
        self.dbias = Some(grad.sum_rows());
        self.dweights = Some(self.inputs.as_ref().unwrap().T().gemm(&grad));
        grad.gemm(&self.weights.T())
    }

    pub fn sgd(&mut self, lr: T) {
        let dweights = self.dweights.as_ref().unwrap();
        let dbias = self.dbias.as_ref().unwrap();

        self.weights -= &dweights.muls(lr);
        self.bias -= &dbias.muls(lr);

        /*
        for (idx, value) in self.weights.as_mut_slice().iter_mut().enumerate() {
            *value -= dweights.as_slice()[idx] * lr;
        }

        for (idx, value) in self.bias.as_mut_slice().iter_mut().enumerate() {
            *value -= dbias.as_slice()[idx] * lr;
        }
        */
    }
}

impl<'a, T: Copy, const I: usize, const O: usize> GetParam<'a, T> for Linear<'a, T, I, O> {
    fn params(&mut self) -> Option<Param<'a, T>> {
        Some(Param::new(
            self.weights.shallow(),
            self.bias.shallow(),
            self.dweights.as_ref().unwrap().shallow(),
            self.dbias.as_ref().unwrap().shallow(),
        ))
    }
}

impl<'a, T, const I: usize, const O: usize> Default for Linear<'a, T, I, O> {
    fn default() -> Self {
        Self {
            weights: Default::default(),
            bias: Default::default(),
            dweights: Default::default(),
            dbias: Default::default(),
            inputs: Default::default(),
        }
    }
}
