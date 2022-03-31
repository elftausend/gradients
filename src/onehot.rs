use custos::{number::Number, Matrix, InternCPU, InternCLDevice, opencl::GenericOCL};
use custos_math::switch_to_cpu_help_s;
use purpur::utils::max;

pub fn onehot<T: Number+purpur::number::Number>(device: &InternCPU, matrix: Matrix<T>) -> Matrix<T> {
    if matrix.cols() == 1 {
        let data = matrix.as_cpu_slice();
    
        let max = max(&data).as_usize()+1;
        let mut onehot = vec![T::default(); matrix.rows()*max];
    
        for (row, idx) in data.iter().enumerate() {
            for i in 0..max {
                if i == idx.as_usize() {
                    onehot[row*max+i] = T::one();
                    //onehot.push(T::one()); //via index
                } else {
                    onehot[row*max+i] = T::zero();
                }
            }
        }

        Matrix::from((device, (data.len(), max), onehot))

    } else {
        panic!("wrong rank");
    }   
}

pub fn onehot_cl<T: GenericOCL+purpur::number::Number>(device: &InternCLDevice, x: Matrix<T>) -> Matrix<T> {
    switch_to_cpu_help_s(device, x, |device, x| onehot(device, x))
}