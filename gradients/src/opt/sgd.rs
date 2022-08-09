use crate::Param;
use custos::{Alloc, CDatatype, CPU};
use custos_math::Matrix;

#[cfg(feature = "opencl")]
use custos::{opencl::enqueue_kernel, CLDevice};

pub struct SGD<'a, T> {
    lr: T,
    weight_momentum: Vec<Matrix<'a, T>>,
    bias_momentum: Vec<Matrix<'a, T>>,
    momentum: T,
}

impl<'a, T: CDatatype> SGD<'a, T> {
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

    pub fn step<D: Alloc<T> + SGDOp<T>>(&mut self, device: &'a D, params: Vec<Param<T>>) {
        if self.momentum > T::zero() {
            if self.weight_momentum.len() < params.len() {
                for param in &params {
                    self.weight_momentum
                        .push(Matrix::new(device, param.weights.dims()));
                    self.bias_momentum
                        .push(Matrix::new(device, param.bias.dims()));
                }
            }
            return device.step_momentum(self, params);
        }
        device.step(self, params)
    }
}

pub trait SGDOp<T: CDatatype> {
    fn step(&self, sgd: &mut SGD<T>, params: Vec<Param<T>>) {
        for mut param in params {
            param.weights -= param.dweights * sgd.lr;
            param.bias -= param.dbias * sgd.lr;
        }
    }
    fn step_momentum(&self, sgd: &mut SGD<T>, params: Vec<Param<T>>);
}

impl<T: CDatatype> SGDOp<T> for CPU {
    fn step_momentum(&self, sgd: &mut SGD<T>, mut params: Vec<Param<T>>) {
        for (layer_idx, param) in params.iter_mut().enumerate() {
            for (idx, w) in param.weights.iter_mut().enumerate() {
                let update = sgd.momentum * sgd.weight_momentum[layer_idx][idx]
                    + param.dweights[idx] * sgd.lr;
                *w -= update;
                sgd.weight_momentum[layer_idx][idx] = update;
            }

            for (idx, b) in param.bias.iter_mut().enumerate() {
                let update =
                    sgd.momentum * sgd.bias_momentum[layer_idx][idx] + param.dbias[idx] * sgd.lr;
                *b -= update;
                sgd.bias_momentum[layer_idx][idx] = update;
            }
        }
    }
}

#[cfg(feature = "opencl")]
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
            enqueue_kernel(
                self,
                &src,
                [param.weights.size(), 0, 0],
                None,
                &[
                    &param.weights,
                    &param.dweights,
                    &sgd.weight_momentum[idx],
                    &sgd.momentum,
                    &sgd.lr,
                ],
            )
            .unwrap();

            enqueue_kernel(
                self,
                &src,
                [param.bias.size(), 0, 0],
                None,
                &[
                    &param.bias,
                    &param.dbias,
                    &sgd.bias_momentum[idx],
                    &sgd.momentum,
                    &sgd.lr,
                ],
            )
            .unwrap();
            /*
            KernelOptions::new(
                self,
                param.weights.as_buf(),
                [param.weights.size(), 0, 0],
                &src,
            )
            .unwrap()
            .add_arg(&param.dweights)
            .add_arg(&sgd.weight_momentum[idx])
            .add_arg(&sgd.momentum)
            .add_arg(&sgd.lr)
            .run()
            .unwrap();

            KernelOptions::new(self, param.bias.as_buf(), [param.bias.size(), 0, 0], &src)
                .unwrap()
                .add_arg(&param.dbias)
                .add_arg(&sgd.bias_momentum[idx])
                .add_arg(&sgd.momentum)
                .add_arg(&sgd.lr)
                .run()
                .unwrap();
            */
        }
    }
}
