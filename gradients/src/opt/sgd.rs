use custos::{opencl::KernelOptions, CDatatype, CLDevice, CPU, Matrix};
use custos_math::Additional;

use crate::Param;

pub struct SGD<T> {
    lr: T,
    weight_momentum: Vec<Matrix<T>>,
    bias_momentum: Vec<Matrix<T>>,
    momentum: T,
}

impl<T: CDatatype> SGD<T> {
    pub fn new(lr: T) -> Self {
        SGD {
            lr,
            weight_momentum: Vec::new(),
            bias_momentum: Vec::new(),
            momentum: T::one() / T::two(),
        }
    }

    pub fn momentum(mut self, momentum: T) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn step<D: SGDOp<T>>(&mut self, device: &D, params: Vec<Param<T>>) {
        if self.momentum > T::zero() {
            if self.weight_momentum.len() < params.len() {
                for param in &params {
                    self.weight_momentum
                        .push(Matrix::from(param.weights.dims()));
                    self.bias_momentum.push(Matrix::from(param.bias.dims()));
                }
            }

            device.step_momentum(self, params)
        } else {
            device.step(self, params)
        }
    }
}

pub trait SGDOp<T: CDatatype> {
    fn step(&self, sgd: &mut SGD<T>, params: Vec<Param<T>>) {
        for mut param in params {
            param.weights -= &param.dweights.muls(sgd.lr);
            param.bias -= &param.dbias.muls(sgd.lr);
        }
    }
    fn step_momentum(&self, sgd: &mut SGD<T>, params: Vec<Param<T>>);
}

impl<T: CDatatype> SGDOp<T> for CPU {
    fn step_momentum(&self, sgd: &mut SGD<T>, mut params: Vec<Param<T>>) {
        for (layer_idx, param) in params.iter_mut().enumerate() {
            for (idx, w) in param.weights.as_mut_slice().iter_mut().enumerate() {
                let update = sgd.momentum * sgd.weight_momentum[layer_idx].as_slice()[idx]
                    + param.dweights.as_slice()[idx] * sgd.lr;
                *w -= update;
                sgd.weight_momentum[layer_idx].as_mut_slice()[idx] = update;
            }

            for (idx, b) in param.bias.as_mut_slice().iter_mut().enumerate() {
                let update = sgd.momentum * sgd.bias_momentum[layer_idx].as_slice()[idx]
                    + param.dbias.as_slice()[idx] * sgd.lr;
                *b -= update;
                sgd.bias_momentum[layer_idx].as_mut_slice()[idx] = update;
            }
        }
    }
}

impl<T: CDatatype> SGDOp<T> for CLDevice {
    fn step_momentum(&self, sgd: &mut SGD<T>, params: Vec<Param<T>>) {
        let src = format!(
            "
            __kernel void sgd_momentum(
                __global {dt}* values, 
                __global const {dt}* dvalues, 
                __global {dt}* value_momentum,
                const {dt} momentum,
                const {dt} lr)
                {{
                    size_t i = get_global_id(0);
                    
                    {dt} value_update = momentum * value_momentum[i] + dvalues[i] * lr;
                    values[i] -= value_update;
                    value_momentum[i] = value_update;
                }}
        ",
            dt = T::as_c_type_str()
        );

        for (idx, param) in params.iter().enumerate() {
            KernelOptions::new(self, param.weights.as_buf(), [param.weights.size(), 0, 0], &src).unwrap()
                .add_arg(&param.dweights)
                .add_arg(&sgd.weight_momentum[idx])
                .add_arg(&sgd.momentum)
                .add_arg(&sgd.lr)
                .run()
                .unwrap();

            KernelOptions::new(self, param.bias.as_buf(), [param.bias.size(), 0, 0], &src).unwrap()
                .add_arg(&param.dbias)
                .add_arg(&sgd.bias_momentum[idx])
                .add_arg(&sgd.momentum)
                .add_arg(&sgd.lr)
                .run()
                .unwrap();
        }
    }
}
