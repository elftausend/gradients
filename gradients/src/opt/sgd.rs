use crate::Param;
use custos_math::custos::{prelude::Number, Alloc, CDatatype, Device, GraphReturn, MainMemory, CPU};
use custos_math::{AdditionalOps, AssignOps, BaseOps, Matrix};

#[cfg(feature = "opencl")]
use custos_math::custos::{opencl::enqueue_kernel, OpenCL};

pub struct SGD<'a, T, D: Device = CPU> {
    lr: T,
    weight_momentum: Vec<Matrix<'a, T, D>>,
    bias_momentum: Vec<Matrix<'a, T, D>>,
    momentum: T,
}

impl<'a, T: Number, D> SGD<'a, T, D>
where
    D: Alloc<'a, T> + SGDOp<T> + GraphReturn,
{
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

    pub fn step(&mut self, device: &'a D, params: Vec<Param<T, D>>) {
        if self.momentum > T::zero() {
            if self.weight_momentum.len() < params.len() {
                for param in &params {
                    self.weight_momentum
                        .push(Matrix::new(device, param.weights.dims()));

                    if let Some(bias) = &param.bias {
                        self.bias_momentum.push(Matrix::new(device, bias.dims()));
                    }
                }
            }
            return device.step_momentum(self, params);
        }
        device.step(self, params)
    }
}

pub trait SGDOp<T: Number, D: Device = Self>: BaseOps<T> + AssignOps<T> + AdditionalOps<T> {
    fn step(&self, sgd: &mut SGD<T, D>, params: Vec<Param<T, D>>)
    where
        D: BaseOps<T> + AssignOps<T> + AdditionalOps<T>,
    {
        for mut param in params {
            param.weights -= param.dweights * sgd.lr;

            if let Some(mut bias) = param.bias {
                bias -= param.dbias * sgd.lr;
            }
        }
    }
    fn step_momentum(&self, sgd: &mut SGD<T, D>, params: Vec<Param<T, D>>);
}

impl<T: Number, D: MainMemory> SGDOp<T, D> for CPU {
    fn step_momentum(&self, sgd: &mut SGD<T, D>, mut params: Vec<Param<T, D>>) {
        for (layer_idx, param) in params.iter_mut().enumerate() {
            for (idx, w) in param.weights.iter_mut().enumerate() {
                let update = sgd.momentum * sgd.weight_momentum[layer_idx][idx]
                    + param.dweights[idx] * sgd.lr;
                *w -= update;
                sgd.weight_momentum[layer_idx][idx] = update;
            }

            if let Some(bias) = &mut param.bias {
                for (idx, b) in bias.iter_mut().enumerate() {
                    let update = sgd.momentum * sgd.bias_momentum[layer_idx][idx]
                        + param.dbias[idx] * sgd.lr;
                    *b -= update;
                    sgd.bias_momentum[layer_idx][idx] = update;
                }
            }
        }
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype + Number> SGDOp<T> for OpenCL {
    fn step_momentum(&self, sgd: &mut SGD<T, Self>, params: Vec<Param<T, Self>>) {
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

            if let Some(bias) = &param.bias {
                enqueue_kernel(
                    self,
                    &src,
                    [bias.size(), 0, 0],
                    None,
                    &[
                        &bias,
                        &param.dbias,
                        &sgd.bias_momentum[idx],
                        &sgd.momentum,
                        &sgd.lr,
                    ],
                )
                .unwrap();
            }
        }
    }
}
