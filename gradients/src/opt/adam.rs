use crate::Param;
use custos::{number::Float, Alloc, CDatatype, Device, GraphReturn, CPU};
use custos_math::Matrix;

#[cfg(feature = "cuda")]
use custos::cuda::launch_kernel1d;

#[cfg(feature = "opencl")]
use custos::{opencl::enqueue_kernel, OpenCL};

pub struct Adam<'a, T, D: Device = CPU> {
    lr: T,
    epsilon: T,
    beta1: T,
    beta2: T,
    pub iters: u64,
    weight_momentum: Vec<Matrix<'a, T, D>>,
    weight_cache: Vec<Matrix<'a, T, D>>,
    bias_momentum: Vec<Matrix<'a, T, D>>,
    bias_cache: Vec<Matrix<'a, T, D>>,
}

impl<'a, T: Float, D> Adam<'a, T, D>
where
    D: Alloc<'a, T> + GraphReturn + AdamOp<'a, T>,
{
    pub fn new(lr: T) -> Adam<'a, T, D> {
        Adam {
            lr,
            epsilon: T::as_generic(1e-7),
            beta1: T::as_generic(0.9),
            beta2: T::as_generic(0.999),
            iters: 0,
            weight_momentum: Vec::new(),
            weight_cache: Vec::new(),
            bias_momentum: Vec::new(),
            bias_cache: Vec::new(),
        }
    }
    pub fn step(&mut self, device: &'a D, params: Vec<Param<'a, T, D>>) {
        if self.weight_cache.len() < params.len() {
            for param in params.iter() {
                self.weight_cache
                    .push(Matrix::new(device, param.weights.dims()));
                self.weight_momentum
                    .push(Matrix::new(device, param.weights.dims()));

                if let Some(bias) = &param.bias {
                    self.bias_cache.push(Matrix::new(device, bias.dims()));
                    self.bias_momentum.push(Matrix::new(device, bias.dims()));
                }
            }
        }
        device.step(self, params);
        self.iters += 1;
    }
}

pub trait AdamOp<'a, T, D: Device = Self> {
    fn step(&'a self, adam: &mut Adam<'a, T, D>, params: Vec<Param<'a, T, D>>);
}

fn adam_step_cpu<T: Float>(
    values: &mut [T],
    dvalues: &[T],
    momentum: &mut [T],
    cache: &mut [T],
    beta1: T,
    beta2: T,
    epsilon: T,
    lr: T,
    iters: u64,
) {
    for i in 0..values.len() {
        momentum[i] = momentum[i] * beta1 + dvalues[i] * (T::one() - beta1);
        let value_momentum_corrected = momentum[i] / (T::one() - beta1.powi((iters + 1) as i32));
        cache[i] = cache[i] * beta2 + dvalues[i] * dvalues[i] * (T::one() - beta2);
        let value_cache_corrected = cache[i] / (T::one() - beta2.powi((iters + 1) as i32));
        values[i] -= (value_momentum_corrected * lr) / (value_cache_corrected.sqrt() + epsilon);
    }
}

impl<'a, T: Float> AdamOp<'a, T> for CPU {
    fn step(&'a self, adam: &mut Adam<'a, T>, mut params: Vec<Param<'a, T>>) {
        for (idx, param) in params.iter_mut().enumerate() {
            adam_step_cpu(
                &mut param.weights,
                &param.dweights,
                &mut adam.weight_momentum[idx],
                &mut adam.weight_cache[idx],
                adam.beta1,
                adam.beta2,
                adam.epsilon,
                adam.lr,
                adam.iters,
            );
            if let Some(bias) = &mut param.bias {
                adam_step_cpu(
                    bias,
                    &param.dbias,
                    &mut adam.bias_momentum[idx],
                    &mut adam.bias_cache[idx],
                    adam.beta1,
                    adam.beta2,
                    adam.epsilon,
                    adam.lr,
                    adam.iters,
                );
            }
        }
    }
}

#[cfg(feature = "cuda")]
impl<'a, T: CDatatype> AdamOp<'a, T> for custos::CudaDevice {
    fn step(&self, adam: &mut Adam<T>, mut params: Vec<Param<T>>) {
        let src = format!(
            r#"extern "C" __global__ void adam(
                {dt}* value, 
                {dt}* dvalue, 
                {dt}* value_momentum, 
                {dt}* value_cache, 
                {dt} beta1,
                {dt} beta2,
                {dt} epsilon,
                {dt} iters,
                {dt} lr,
                int numElements
            )
                {{
                    int idx = blockDim.x * blockIdx.x + threadIdx.x;
                    if (idx < numElements) {{
                        value_momentum[idx] = value_momentum[idx] * beta1 + dvalue[idx] * (1.0-beta1);
                        {dt} value_momentum_corrected = value_momentum[idx] / (1.0 - powf(beta1, iters + 1));
                        value_cache[idx] = value_cache[idx]*beta2 + dvalue[idx] * dvalue[idx] * (1.0-beta2);
                        {dt} value_cache_corrected = value_cache[idx] / (1.0 - powf(beta2, iters + 1));                            
                        value[idx] -= (value_momentum_corrected * lr) / (sqrtf(value_cache_corrected) + epsilon);
                    }}
                  
                }}
        "#,
            dt = T::as_c_type_str()
        );

        for (idx, layer_data) in params.iter_mut().enumerate() {
            launch_kernel1d(
                layer_data.weights.size(),
                self,
                &src,
                "adam",
                &[
                    &layer_data.weights.as_buf(),
                    &layer_data.dweights.as_buf(),
                    &adam.weight_momentum[idx].as_buf(),
                    &adam.weight_cache[idx].as_buf(),
                    &adam.beta1,
                    &adam.beta2,
                    &adam.epsilon,
                    &adam.iters,
                    &adam.lr,
                    &layer_data.weights.size(),
                ],
            )
            .unwrap();

            launch_kernel1d(
                layer_data.bias.size(),
                self,
                &src,
                "adam",
                &[
                    &layer_data.bias.as_buf(),
                    &layer_data.dbias.as_buf(),
                    &adam.bias_momentum[idx].as_buf(),
                    &adam.bias_cache[idx].as_buf(),
                    &adam.beta1,
                    &adam.beta2,
                    &adam.epsilon,
                    &adam.iters,
                    &adam.lr,
                    &layer_data.bias.size(),
                ],
            )
            .unwrap();
        }
    }
}

#[cfg(feature = "opencl")]
impl<'a, T: CDatatype> AdamOp<'a, T> for OpenCL {
    fn step(&self, adam: &mut Adam<T, Self>, mut params: Vec<Param<T, Self>>) {
        let src = format!("__kernel void adam(
            __global {dt}* value, 
            __global const {dt}* dvalue, 
            __global {dt}* value_momentum,
            __global {dt}* value_cache,
            const {dt} beta1,
            const {dt} beta2,
            const {dt} epsilon,
            const long iters,
            const {dt} lr) 
            
            {{
                size_t idx = get_global_id(0);
                
                value_momentum[idx] = value_momentum[idx] * beta1 + dvalue[idx] * (1.0-beta1);
            
                {dt} value_momentum_corrected = value_momentum[idx] / (1.0 - pow(beta1, iters + 1));
                value_cache[idx] = value_cache[idx]*beta2 + pow(dvalue[idx], 2) * (1.0-beta2);
                {dt} value_cache_corrected = value_cache[idx] / (1.0 - pow(beta2, iters + 1));                            
                value[idx] -= (value_momentum_corrected * lr) / (sqrt(value_cache_corrected) + epsilon);
            }}", dt=T::as_c_type_str());

        for (idx, layer_data) in params.iter_mut().enumerate() {
            enqueue_kernel(
                self,
                &src,
                [layer_data.weights.size(), 0, 0],
                None,
                &[
                    &layer_data.weights,
                    &layer_data.dweights,
                    &adam.weight_momentum[idx],
                    &adam.weight_cache[idx],
                    &adam.beta1,
                    &adam.beta2,
                    &adam.epsilon,
                    &(adam.iters + 1),
                    &adam.lr,
                ],
            )
            .unwrap();

            if let Some(bias) = &layer_data.bias {
                enqueue_kernel(
                    self,
                    &src,
                    [bias.size(), 0, 0],
                    None,
                    &[
                        bias,
                        &layer_data.dbias,
                        &adam.bias_momentum[idx],
                        &adam.bias_cache[idx],
                        &adam.beta1,
                        &adam.beta2,
                        &adam.epsilon,
                        &(adam.iters + 1),
                        &adam.lr,
                    ],
                )
                .unwrap();
            }
        }
    }
}
