use crate::Param;
use custos::{number::Float, CDatatype, CPU, Alloc};
use custos_math::{scalar_apply, AdditionalOps, BaseOps, Matrix};

#[cfg(feature = "cuda")]
use custos::cuda::launch_kernel1d;

#[cfg(feature = "opencl")]
use custos::{CLDevice, opencl::enqueue_kernel};

pub struct Adam<'a, T> {
    lr: T,
    epsilon: T,
    beta1: T,
    beta2: T,
    pub iters: u64,
    weight_momentum: Vec<Matrix<'a, T>>,
    weight_cache: Vec<Matrix<'a, T>>,
    bias_momentum: Vec<Matrix<'a, T>>,
    bias_cache: Vec<Matrix<'a, T>>,
}

impl<'a, T: Float> Adam<'a, T> {
    pub fn new(lr: T) -> Adam<'a, T> {
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
    pub fn step<D: Alloc<T> + AdamOp<'a, T>>(&mut self, device: &'a D, params: Vec<Param<T>>) {
        if self.weight_cache.len() < params.len() {
            for param in params.iter() {
                self.weight_cache
                    .push(Matrix::new(device, param.weights.dims()));
                self.weight_momentum
                    .push(Matrix::new(device, param.weights.dims()));

                self.bias_cache.push(Matrix::new(device, param.bias.dims()));
                self.bias_momentum
                    .push(Matrix::new(device, param.bias.dims()));
            }
        }
        device.step(self, params);
    }
}

pub trait AdamOp<'a, T> {
    fn step(&'a self, adam: &mut Adam<'a, T>, params: Vec<Param<T>>);
}

impl<'a, T: CDatatype + Float> AdamOp<'a, T> for CPU {
    fn step(&'a self, adam: &mut Adam<'a, T>, mut params: Vec<Param<T>>) {
        for (idx, param) in params.iter_mut().enumerate() {
            adam.weight_momentum[idx] = self.muls(&adam.weight_momentum[idx], adam.beta1)
                + self.muls(&param.dweights, T::one() - adam.beta1);
            adam.bias_momentum[idx] = self.muls(&adam.bias_momentum[idx], adam.beta1)
                + self.muls(&param.dbias, T::one() - adam.beta1);

            let weight_momentum_corrected = self.divs(
                &adam.weight_momentum[idx],
                T::one() - adam.beta1.powi((adam.iters as i32) + 1),
            );
            let bias_momentum_corrected = self.divs(
                &adam.bias_momentum[idx],
                T::one() - adam.beta1.powi((adam.iters as i32) + 1),
            );

            let map_dweights = scalar_apply(self, &param.dweights, T::zero(), |c, a, _| {
                *c = a.powi(2) * (T::one() - adam.beta2)
            });
            adam.weight_cache[idx] = self.muls(&adam.weight_cache[idx], adam.beta2) + map_dweights;

            let map_dbias = scalar_apply(self, &param.dbias, T::zero(), |c, a, _| {
                *c = a.powi(2) * (T::one() - adam.beta2)
            });

            adam.bias_cache[idx] = (&adam.bias_cache[idx] * adam.beta2) + map_dbias;
            //adam.bias_cache[idx] = self.muls(&adam.bias_cache[idx], adam.beta2) + map_dbias;

            let weight_cache_corrected = self.divs(
                &adam.weight_cache[idx],
                T::one() - adam.beta2.powi((adam.iters as i32) + 1),
            );
            let bias_cache_corrected = self.divs(
                &adam.bias_cache[idx],
                T::one() - adam.beta2.powi((adam.iters as i32) + 1),
            );

            let map_weight_cache_corrected =
                scalar_apply(self, &weight_cache_corrected, T::zero(), |c, a, _| {
                    *c = a.sqrt() + adam.epsilon
                });
            param.weights -= &self.div(
                &self.muls(&weight_momentum_corrected, adam.lr),
                &map_weight_cache_corrected,
            );

            let map_bias_cache_corrected =
                scalar_apply(self, &bias_cache_corrected, T::zero(), |c, a, _| {
                    *c = a.sqrt() + adam.epsilon
                });
            param.bias -= &self.div(
                &self.muls(&bias_momentum_corrected, adam.lr),
                &map_bias_cache_corrected,
            );
        }
        let iters = &mut adam.iters;
        *iters += 1;
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
                vec![
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
                vec![
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
impl<'a, T: CDatatype> AdamOp<'a, T> for CLDevice {
    fn step(&self, adam: &mut Adam<T>, mut params: Vec<Param<T>>) {
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
            enqueue_kernel(self, &src, [layer_data.weights.size(), 0, 0], None, &[
                &layer_data.weights, &layer_data.dweights, &adam.weight_momentum[idx], &adam.weight_cache[idx], &adam.beta1,
                &adam.beta2, &adam.epsilon, &(adam.iters + 1), &adam.lr,
            ]).unwrap();

            enqueue_kernel(self, &src, [layer_data.bias.size(), 0, 0], None, &[
                &layer_data.bias, &layer_data.dbias, &adam.bias_momentum[idx], &adam.bias_cache[idx], &adam.beta1,
                &adam.beta2, &adam.epsilon, &(adam.iters + 1), &adam.lr,
            ]).unwrap();
        }
    }
}
