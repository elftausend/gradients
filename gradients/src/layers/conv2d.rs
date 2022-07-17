use custos::{cached, number::Float, CDatatype};
use custos_math::{assign_to_lhs, Matrix};

pub struct KernelBlock<T> {
    pub weights: Matrix<T>,
    bias: Matrix<T>,
    //dweights: Matrix<T>
}

impl<T> KernelBlock<T> {
    pub fn new(shape: (usize, usize), bias_shape: (usize, usize)) -> Self
    where
        T: Float,
    {
        let mut weights = Matrix::from(shape);
        weights.rand(T::one().neg(), T::one());

        let mut bias = Matrix::from(bias_shape);
        bias.rand(T::one().neg(), T::one());

        KernelBlock { weights, bias }
    }
}

pub struct Conv2D<T> {
    pub kernel_shape: (usize, usize),
    output_shape: (usize, usize),
    kernels: Vec<KernelBlock<T>>,
}

impl<T: Float + CDatatype> Conv2D<T> {
    pub fn new(
        input_shape: (usize, usize),
        kernel_shape: (usize, usize),
        kernel_blocks: usize,
    ) -> Conv2D<T> {
        let output_shape = (
            input_shape.0 - kernel_shape.0 + 1,
            input_shape.1 - kernel_shape.1 + 1,
        );
        let kernels = (0..kernel_blocks)
            .into_iter()
            .map(|_x| KernelBlock::new(kernel_shape, output_shape))
            .collect();

        Conv2D {
            kernel_shape,
            output_shape,
            kernels,
        }
    }

    pub fn forward(&mut self, _inputs: Matrix<T>) -> Matrix<T> {
        let (out_rows, out_cols) = self.output_shape;

        /*let mut output = cached::<T>(out_rows*out_cols * self.kernels.len());
        // TODO: use set_.. with bias instead
        output.clear();
        for kernel_block in &self.kernels {
            output += kernel_block.bias;
        }*/
        let mut output = cached::<T>(out_rows * out_cols * self.kernels.len());

        for (idx, kernel_block) in self.kernels.iter().enumerate() {
            let start = idx * out_rows * out_cols;
            let output_slice = &mut output[start..start + out_rows * out_cols];
            assign_to_lhs(output_slice, &kernel_block.bias, |a, b| *a += b);

            //correlate_add(output_slice, inputs, kernel_block.weights)
        }
        //output
        todo!()
    }
}
