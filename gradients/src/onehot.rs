use custos::{number::Number, CDatatype, Device, MainMemory, Shape, CPU};
use custos_math::Matrix;
use purpur::utils::max;

#[cfg(feature = "opencl")]
use custos_math::cl_to_cpu_s;

pub trait OneHotMat<'a, T, D: Device> {
    fn onehot(&self) -> Matrix<'a, T, D>;
}

impl<'a, T: Number + CDatatype, D: OnehotOp<T>> OneHotMat<'a, T, D> for Matrix<'a, T, D> {
    fn onehot(&self) -> Matrix<'a, T, D> {
        self.device().onehot(self)
    }
}

pub trait OnehotOp<T, IS: Shape = (), OS: Shape = (), D: Device = Self>: Device {
    fn onehot(&self, matrix: &Matrix<T, D, IS>) -> Matrix<T, Self, OS>;
}

impl<T: Number, IS: Shape, OS: Shape, D: MainMemory> OnehotOp<T, IS, OS, D> for CPU {
    fn onehot(&self, matrix: &Matrix<T, D, IS>) -> Matrix<T, Self, OS> {
        assert!(matrix.cols() == 1);

        let max = max(matrix).as_usize() + 1;
        let mut onehot = self.retrieve(matrix.rows() * max, matrix.as_buf());

        for (row, idx) in matrix.iter().enumerate() {
            for i in 0..max {
                if i == idx.as_usize() {
                    onehot[row * max + i] = T::one();
                } else {
                    onehot[row * max + i] = T::zero();
                }
            }
        }

        (onehot, matrix.rows(), max).into()
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> OnehotOp<T> for custos::OpenCL {
    fn onehot(&self, x: &Matrix<T, Self>) -> Matrix<T, Self> {
        cl_to_cpu_s(self, x, |device, x| device.onehot(x))
    }
}

#[cfg(feature = "cuda")]
impl<T: CDatatype> OnehotOp<T> for custos::CudaDevice {
    fn onehot(&self, x: &Matrix<T>) -> Matrix<T> {
        custos_math::cu_to_cpu_s(self, x, |device, x| device.onehot(x))
    }
}
