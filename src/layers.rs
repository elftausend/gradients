use custos::{Matrix, cpu::TBlas, number::Float, get_device};

#[derive(Clone, Copy)]
pub struct Linear<T> {
    pub weights: Matrix<T>,
    pub bias: Matrix<T>,
    pub dweights: Option<Matrix<T>>,
    pub dbias: Option<Matrix<T>>,
    inputs: Option<Matrix<T>>,
}

impl <T: Float+TBlas>Linear<T> {
    pub fn new(input_size: usize, output_size: usize, weight_size: T) -> Linear<T> {
        let device = get_device!();
        let mut weights = Matrix::<T>::new((input_size, output_size));
        
        weights.rand();
        let weights = weights * weight_size;
        //let weights = weights + (T::one() / T::from_usize(100));

        let bias = Matrix::<T>::new((1, output_size));

        Linear { weights, bias, dweights: None, dbias: None, inputs: None }
    }
    pub fn set_weights(&mut self, weights: &[T]) {
        self.weights.copy_from_slice(weights)
    }

    pub fn forward(&mut self, inputs: Matrix<T>) -> Matrix<T> {
        self.inputs = Some(inputs);
        inputs.gemm(self.weights).row_op(self.bias, |x, b| x+b)
    }

    pub fn backward(&mut self, grad: Matrix<T>) -> Matrix<T> {
        self.dbias = Some(grad.sum_axis(Axis::Rows));
        self.dweights = Some(self.inputs.unwrap().T().gemm(grad));
        grad.gemm(self.weights.T())
    }

    pub fn sgd(&mut self, lr: T) {
        let dweights = self.dweights.unwrap();
        let dbias = self.dbias.unwrap();
        for (idx, value) in self.weights.as_mut_slice().iter_mut().enumerate() {
            *value -= dweights.as_slice()[idx] * lr;
        }

        for (idx, value) in self.bias.as_mut_slice().iter_mut().enumerate() {
            *value -= dbias.as_slice()[idx] * lr;
        }
        
    }
}

#[derive(Clone, Copy)]
pub struct ReLU<T> {
    inputs: Option<Matrix<T>>
}

impl <T: Float>ReLU<T> {
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



