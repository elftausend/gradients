use custos::{number::Number, CDatatype, InternCLDevice, InternCPU, Matrix};
use custos_math::cl_to_cpu_s;
use purpur::utils::max;

pub trait OnehotOp<T> {
    fn onehot(&self, matrix: Matrix<T>) -> Matrix<T>;
}

impl<T: Number> OnehotOp<T> for InternCPU {
    fn onehot(&self, matrix: Matrix<T>) -> Matrix<T> {
        assert!(matrix.cols() == 1);

        let data = matrix.as_slice();

        let max = max(data).as_usize() + 1;
        let mut onehot = vec![T::default(); matrix.rows() * max];

        for (row, idx) in data.iter().enumerate() {
            for i in 0..max {
                if i == idx.as_usize() {
                    onehot[row * max + i] = T::one();
                } else {
                    onehot[row * max + i] = T::zero();
                }
            }
        }

        Matrix::from((self, (data.len(), max), onehot))
    }
}

impl<T: CDatatype> OnehotOp<T> for InternCLDevice {
    fn onehot(&self, x: Matrix<T>) -> Matrix<T> {
        cl_to_cpu_s(self, &x, |device, x| device.onehot(x))
    }
}
