use custos::{Device, Matrix, InternCPU, number::Float, InternCLDevice, VecRead, opencl::cl_write, get_device};
use rand::{thread_rng, Rng};

pub trait RandMatrix {
    fn rand(&mut self);
}
impl <T: Float>RandMatrix for Matrix<T> {
    fn rand(&mut self) {
        let device = get_device!(RandOp, T).unwrap();
        device.rand(self)
    }
}

pub trait RandOp<T>: Device<T> {
    fn rand(&self, x: &mut Matrix<T>);
}

impl<T: Float> RandOp<T> for InternCPU {
    fn rand(&self, x: &mut Matrix<T>) {
        let mut rng = thread_rng();
        for value in x.as_cpu_slice_mut() {
            *value = rng.gen_range(T::one().negate()..T::one());
        }
    }
}

impl<T: Float> RandOp<T> for InternCLDevice {
    fn rand(&self, x: &mut Matrix<T>) {
        let mut rng = thread_rng();
        let mut data = self.read(x.data());
        
        for value in data.iter_mut() {
            *value = rng.gen_range(T::one().negate()..T::one());
        }
        cl_write(self, x, &data);
    }
}