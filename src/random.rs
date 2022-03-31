use custos::{Device, Matrix, InternCPU, number::Float, InternCLDevice, VecRead, opencl::cl_write};
use rand::{thread_rng, Rng};

pub trait RandOp<T>: Device<T> {
    fn rand(&self, x: &mut Matrix<T>);
}

impl <T: Float>RandOp<T> for InternCPU {
    fn rand(&self, x: &mut Matrix<T>) {
        let mut rng = thread_rng();
        for value in x.as_cpu_slice_mut() {
            *value = rng.gen_range(T::one().negate()..T::one());
        }
    }
}

impl <T: Float>RandOp<T> for InternCLDevice {
    fn rand(&self, x: &mut Matrix<T>) {
        let mut rng = thread_rng();
        let mut data = self.read(x.data());
        
        for value in data.iter_mut() {
            *value = rng.gen_range(T::one().negate()..T::one());
        }
        cl_write(self, x, &data);
    }
}