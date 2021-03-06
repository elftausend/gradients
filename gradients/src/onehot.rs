use custos::{number::Number, CPU};
use custos_math::Matrix;
use purpur::utils::max;

#[cfg(any(feature = "opencl", feature = "cuda"))]
use custos::CDatatype;
#[cfg(feature = "opencl")]
use custos_math::cl_to_cpu_s;

pub trait OnehotOp<T> {
    fn onehot(&self, matrix: Matrix<T>) -> Matrix<T>;
}

impl<T: Number> OnehotOp<T> for CPU {
    fn onehot(&self, matrix: Matrix<T>) -> Matrix<T> {
        assert!(matrix.cols() == 1);

        let max = max(&matrix).as_usize() + 1;
        let mut onehot = Matrix::new(self, (matrix.rows(), max));

        for (row, idx) in matrix.iter().enumerate() {
            for i in 0..max {
                if i == idx.as_usize() {
                    onehot[row * max + i] = T::one();
                } else {
                    onehot[row * max + i] = T::zero();
                }
            }
        }

        onehot
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> OnehotOp<T> for custos::CLDevice {
    fn onehot(&self, x: Matrix<T>) -> Matrix<T> {
        cl_to_cpu_s(self, &x, |device, x| device.onehot(x))
    }
}

#[cfg(feature = "cuda")]
impl<T: CDatatype> OnehotOp<T> for custos::CudaDevice {
    fn onehot(&self, x: Matrix<T>) -> Matrix<T> {
        custos_math::cu_to_cpu_s(self, &x, |device, x| device.onehot(x))
    }
}
