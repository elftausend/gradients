use custos::{cached, number::Float, CDatatype};
use custos_math::{assign_to_lhs, Matrix, correlate_valid_mut};
use gradients_derive::NoParams;
use crate::GetParam;

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

#[derive(NoParams)]
pub struct Conv2D<T> {
    pub kernel_shape: (usize, usize),
    output_shape: (usize, usize),
    kernels: Vec<KernelBlock<T>>,
    inputs: Option<Matrix<T>>
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
            inputs: None
        }
    }

    pub fn forward(&mut self, inputs: Matrix<T>) -> Matrix<T> {
        self.inputs = Some(inputs);
        let (out_rows, out_cols) = self.output_shape;

        let mut output = cached::<T>(out_rows * out_cols * self.kernels.len());
        output.clear();

        for (idx, kernel_block) in self.kernels.iter().enumerate() {
            let start = idx * out_rows * out_cols;
            let output_slice = &mut output[start..start + out_rows * out_cols];
            assign_to_lhs(output_slice, &kernel_block.bias, |a, b| *a += b);
            
            correlate_valid_mut(
                &inputs,
                inputs.dims(),
                &kernel_block.weights,
                kernel_block.weights.dims(),
                output_slice,
            );
        }
        (output, (1, output.len())).into()
    }
    pub fn backward(&mut self, grad: Matrix<T>) -> Matrix<T> {
        let (out_rows, out_cols) = self.output_shape;
        let (kernel_rows, kernel_cols) = self.kernel_shape;
        let mut dkernel = cached::<T>(kernel_rows*kernel_cols*self.kernels.len());
        dkernel.clear();

        for (idx, kernel) in self.kernels.iter_mut().enumerate() {
            let start = idx * out_rows * out_cols;
            let grad_slice = &grad[start..start + out_rows * out_cols];

            let start = idx * kernel_rows * kernel_cols;
            let dkernel_slice = &mut dkernel[start..start+kernel_rows*kernel_cols];

            let inputs = self.inputs.unwrap();
            correlate_valid_mut(&inputs, inputs.dims(), grad_slice, (out_rows, out_cols), dkernel_slice);
            
            // step
            for (idx, value) in kernel.weights.iter_mut().enumerate() {
                *value -= dkernel_slice[idx] * T::one() / T::from_u64(1000);
            }
        }
        // need to calculate w. r. t. inputs
        grad
    }
}

impl<T> Default for Conv2D<T> {
    fn default() -> Self {
        Self {
            inputs: Default::default(),
            kernel_shape: Default::default(),
            output_shape: Default::default(),
            kernels: Default::default()
        }
    }
}