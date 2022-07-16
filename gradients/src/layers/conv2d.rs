use custos::{number::Float, cached};
use custos_math::Matrix;
use rand::distributions::uniform::SampleUniform;

pub struct KernelBlock<T> {
    weights: Matrix<T>,
    bias: Matrix<T>,
    //dweights: Matrix<T>
}

impl<T> KernelBlock<T> {
    pub fn new(shape: (usize, usize)) -> Self where T: Float+SampleUniform {
        let mut weights = Matrix::from(shape);
        weights.rand(T::one().neg(), T::one());

        let mut bias = Matrix::from(shape);
        bias.rand(T::one().neg(), T::one());

        KernelBlock {
            weights,
            bias
        }
    }
}

pub struct Conv2D<T> {
    kernel_shape: (usize, usize),
    kernels: Vec<KernelBlock<T>>,
}

impl<T: Float+SampleUniform> Conv2D<T> {
    pub fn new(kernel_shape: (usize, usize), kernel_blocks: usize) -> Conv2D<T> {
        let kernels = (0..kernel_blocks).into_iter()
            .map(|_x| KernelBlock::new(kernel_shape)).collect();

        Conv2D {
            kernel_shape,
            kernels
        }
    }

    pub fn forward(&mut self, inputs: Matrix<T>) {
        let (out_rows, out_cols) = (inputs.rows() - self.kernel_shape.0, inputs.cols() - self.kernel_shape.1);
        let _output = cached::<T>(out_rows*out_cols);
        
    }
}
