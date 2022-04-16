use custos::{Device, Matrix, InternCPU, number::Float, InternCLDevice, VecRead, opencl::cl_write, get_device};
use rand::{thread_rng, Rng};

pub trait RandMatrix<T> {
    fn rand(&mut self, lo: T, hi: T);
}
impl<T: Float> RandMatrix<T> for Matrix<T> {
    fn rand(&mut self, lo: T, hi: T) {
        let device = get_device!(RandOp, T).unwrap();
        device.rand(self, lo, hi)
    }
}

pub trait RandOp<T>: Device<T> {
    fn rand(&self, x: &mut Matrix<T>, lo: T, hi: T);
}

impl<T: Float> RandOp<T> for InternCPU {
    fn rand(&self, x: &mut Matrix<T>, lo: T, hi: T) {
        let mut rng = thread_rng();
        for value in x.as_cpu_slice_mut() {
            *value = rng.gen_range(lo..hi);
        }
    }
}

impl<T: Float> RandOp<T> for InternCLDevice {
    fn rand(&self, x: &mut Matrix<T>, lo: T, hi: T) {
        let mut rng = thread_rng();
        let mut data = self.read(x.data());
        
        for value in data.iter_mut() {
            *value = rng.gen_range(lo..hi);
        }
        cl_write(self, x, &data);
    }
}