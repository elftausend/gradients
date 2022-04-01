use custos::{Matrix, cpu::TBlas, number::Float, opencl::GenericOCL};
use custos_math::{Additional, Row, Sum, Transpose, nn::Activations};
use crate::RandMatrix;

#[derive(Clone, Copy)]
pub struct Linear<T> {
    pub weights: Matrix<T>,
    pub bias: Matrix<T>,
    pub dweights: Option<Matrix<T>>,
    pub dbias: Option<Matrix<T>>,
    inputs: Option<Matrix<T>>,
}

impl <T: Float+TBlas+GenericOCL>Linear<T> {
    pub fn new(input_size: usize, output_size: usize, weight_size: T) -> Linear<T> {
        let mut weights = Matrix::<T>::from((input_size, output_size));
        
        weights.rand();
        
        let weights = weights.muls(weight_size);
        //let weights = weights + (T::one() / T::from_usize(100));

        let bias = Matrix::<T>::from((1, output_size));

        Linear { weights, bias, dweights: None, dbias: None, inputs: None }
    }
    
    pub fn forward(&mut self, inputs: Matrix<T>) -> Matrix<T> {
        self.inputs = Some(inputs);
        inputs.gemm(self.weights).add_row(self.bias)
    }

    pub fn backward(&mut self, grad: Matrix<T>) -> Matrix<T> {
        self.dbias = Some(grad.sum_rows());
        self.dweights = Some(self.inputs.unwrap().T().gemm(grad));
        grad.gemm(self.weights.T())
    }

    pub fn sgd(&mut self, lr: T) {
        let dweights = self.dweights.unwrap();
        let dbias = self.dbias.unwrap();

        self.weights -= dweights.muls(lr);
        self.bias -= dbias.muls(lr);

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

#[derive(Clone, Copy)]
pub struct ReLU<T> {
    inputs: Option<Matrix<T>>
}

impl <T: Float+GenericOCL>ReLU<T> {
    pub fn new() -> ReLU<T> {
        ReLU {
            inputs: None,
        }
    }
    pub fn forward(&mut self, inputs: Matrix<T>) -> Matrix<T> {
        self.inputs = Some(inputs);
        inputs.relu()
    }
    pub fn backward(&self, grad: Matrix<T>) -> Matrix<T> {
        self.inputs.unwrap().relu_grad() * grad
    }
}

/*
#[derive(Clone, Copy)]
pub struct Tanh<T> {
    inputs: Option<Matrix<T>>
}

impl <T: Float>Tanh<T> {
    pub fn new() -> Tanh<T> {
        Tanh {
            inputs: None,
        }
    }
    pub fn forward(&mut self, inputs: Matrix<T>) -> Matrix<T> {
        self.inputs = Some(inputs);
        inputs.tanh()
    }
    pub fn backward(&self, grad: Matrix<T>) -> Matrix<T> {
        self.inputs.unwrap().tanh_grad() * grad
    }
}
*/


