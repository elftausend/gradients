use custos::{number::Float, CDatatype, GenericBlas, Alloc};
use custos_math::{CudaTranspose, Matrix};

use crate::{GetParam, Param};


pub struct Linear<'a, T> {
    pub weights: Matrix<'a, T>,
    pub bias: Matrix<'a, T>,
    pub dweights: Option<Matrix<'a, T>>,
    pub dbias: Option<Matrix<'a, T>>,
    inputs: Option<Matrix<'a, T>>,
}

impl<'a:, T: Float + GenericBlas + CDatatype> Linear<'a, T> {
    pub fn new<D: Alloc<T>>(device: &'a D, input_size: usize, output_size: usize) -> Linear<'a, T> {
        let mut weights = Matrix::<T>::from((device, input_size, output_size));

        let glorot = (T::from_usize(6) / T::from_usize(input_size + output_size)).sqrt();
        weights.rand(glorot.negate(), glorot);

        //let weights = weights.muls(weight_size);
        //let weights = weights + (T::one() / T::from_usize(100));

        let bias = Matrix::<T>::from((device, 1, output_size));

        Linear {
            weights,
            bias,
            dweights: None,
            dbias: None,
            inputs: None,
        }
    }

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

impl<'a, T: Copy> GetParam<'a, T> for Linear<'a, T> {
    fn params(&mut self) -> Option<Param<'a, T>> {
        Some(Param::new(
            self.weights.shallow(),
            self.bias.shallow(),
            self.dweights.as_ref().unwrap().shallow(),
            self.dbias.as_ref().unwrap().shallow()
        ))
    }
}

impl<'a, T> Default for Linear<'a, T> {
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
