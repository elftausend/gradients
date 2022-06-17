use custos::{
    number::Float, opencl::{KernelOptions}, AssignOps, BaseOps, Device, CDatatype, InternCLDevice,
    InternCPU, Matrix,
};
use custos_math::{scalar_apply, AdditionalOps};

use crate::Param;

pub struct Adam<T> {
    lr: T,
    epsilon: T,
    beta1: T,
    beta2: T,
    pub iters: u64,
    weight_momentum: Vec<Matrix<T>>,
    weight_cache: Vec<Matrix<T>>,
    bias_momentum: Vec<Matrix<T>>,
    bias_cache: Vec<Matrix<T>>,
}

impl<T: Float> Adam<T> {
    pub fn new(lr: T) -> Adam<T> {
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
    pub fn step<D: Device<T> + AdamOp<T> + Clone>(&mut self, device: &D, params: Vec<Param<T>>) {
        if self.weight_cache.len() < params.len() {
            for param in params.iter() {
                self.weight_cache
                    .push(Matrix::new(device, param.weights.dims()));
                self.weight_momentum
                    .push(Matrix::new(device, param.weights.dims()));

                self.bias_cache
                    .push(Matrix::new(device, param.bias.dims()));
                self.bias_momentum
                    .push(Matrix::new(device, param.bias.dims()));
            }
        }
        device.step(self, params);
    }
}

pub trait AdamOp<T> {
    fn step(&self, adam: &mut Adam<T>, params: Vec<Param<T>>);
}

impl<T: CDatatype + Float> AdamOp<T> for InternCPU {
    fn step(&self, adam: &mut Adam<T>, mut params: Vec<Param<T>>) {
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

            adam.bias_cache[idx] = self.muls(&adam.bias_cache[idx], adam.beta2) + map_dbias;
            adam.bias_cache[idx] = self.muls(&adam.bias_cache[idx], adam.beta2) + map_dbias;

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

impl<T: CDatatype> AdamOp<T> for InternCLDevice {
    fn step(&self, adam: &mut Adam<T>, mut params: Vec<Param<T>>) {
        let src = format!("__kernel void adam(
            __global {dt}* value, 
            __global const {dt}* dvalue, 
            __global {dt}* value_momentum,
            __global {dt}* value_cache,
            const {dt} beta1,
            const {dt} beta2,
            const {dt} epsilon,
            const {dt} iters,
            const {dt} lr, 
            __global {dt}* output) 
            
            {{
                size_t idx = get_global_id(0);
                
                value_momentum[idx] = value_momentum[idx] * beta1 + dvalue[idx] * (1.0-beta1);
            
                {dt} value_momentum_corrected = value_momentum[idx] / (1.0 - pow(beta1, iters + 1));
                value_cache[idx] = value_cache[idx]*beta2 + pow(dvalue[idx], 2) * (1.0-beta2);
                {dt} value_cache_corrected = value_cache[idx] / (1.0 - pow(beta2, iters + 1));                            
                output[idx] = (value_momentum_corrected * lr) / (sqrt(value_cache_corrected) + epsilon);
            }}", dt=T::as_c_type_str());

        for (idx, layer_data) in params.iter_mut().enumerate() {
            let gws = [layer_data.weights.size(), 0, 0];
            let output = KernelOptions::new(
                    self, 
                    layer_data.weights.as_mut_buf(), 
                    gws, 
                    &src,
                ).unwrap()
                .add_arg(&mut layer_data.dweights)
                .add_arg(&mut adam.weight_momentum[idx])
                .add_arg(&mut adam.weight_cache[idx])
                .add_arg(&mut adam.beta1)
                .add_arg(&mut adam.beta2)
                .add_arg(&mut adam.epsilon)
                .add_arg(&T::from_u64(adam.iters + 1))
                .add_arg(&mut adam.lr)
                .with_output(gws[0])
                .run()
                .unwrap();

            let dims = layer_data.weights.dims();
            self.sub_assign(&mut layer_data.weights, &(output, dims).into());

            let output =
                KernelOptions::new(self, layer_data.bias.as_buf(), [layer_data.bias.size(), 0, 0], &src).unwrap()
                    .add_arg(&layer_data.dbias)
                    .add_arg(&adam.bias_momentum[idx])
                    .add_arg(&adam.bias_cache[idx])
                    .add_arg(&adam.beta1)
                    .add_arg(&adam.beta2)
                    .add_arg(&adam.epsilon)
                    .add_arg(&T::from_u64(adam.iters + 1))
                    .add_arg(&adam.lr)
                    .with_output(layer_data.bias.size())
                    .run()
                    .unwrap();

            let dims = layer_data.bias.dims();
            self.sub_assign(&mut layer_data.bias, &(output, dims).into());
        }
    }
}
