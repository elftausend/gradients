use std::marker::PhantomData;

use gradients::{
    number::{Float, Number}, prelude::ActivationOps, AdditionalOps, AssignOps, CDatatype, CloneBuf, Device,
    Dim2, Gemm, Matrix, MayDim2, RowOp, ShallowCopy, Shape, Stack, SumOps, ToDim, TransposeOp, CPU, CudaTranspose,
};

pub trait Activation<'a, T, D: Device, S: Shape> {
    fn forward(inputs: Matrix<'a, T, D, S>) -> Matrix<'a, T, D, S>;
}

impl<'a, T, D: Device + 'a, S: Shape> Activation<'a, T, D, S> for () {
    #[inline]
    fn forward(inputs: Matrix<'a, T, D, S>) -> Matrix<'a, T, D, S> {
        inputs
    }
}

pub struct ReLU;

impl<'a, T, D: ActivationOps<T, S>, S: Shape> Activation<'a, T, D, S> for ReLU {
    #[inline]
    fn forward(mut inputs: Matrix<'a, T, D, S>) -> Matrix<'a, T, D, S> {
        inputs.relu_mut();
        inputs
    }
}

#[rustfmt::skip]
pub struct Linear2<'a, T, const I: usize, const O: usize, D = CPU, A = (), const SAMPLES: usize=1> 
where 
    D: Device, 
    A: Activation<'a, T, D, Dim2<SAMPLES, O>>
{
    weights: Matrix<'a, T, D, Dim2<I, O>>,
    bias: Option<Matrix<'a, T, D, Dim2<1, O>>>,
    inputs: Option<Matrix<'a, T, D, Dim2<SAMPLES, I>>>,
    activation_inputs: Option<Matrix<'a, T, D, Dim2<SAMPLES, O>>>,
    dweights: Option<Matrix<'a, T, D>>,
    dbias: Option<Matrix<'a, T, D>>,
    pub l2_reg: T,
    _p: PhantomData<A>
}

impl<'a, T, const I: usize, const O: usize, D, A, const SAMPLES: usize>
    Linear2<'a, T, I, O, D, A, SAMPLES>
where
    T: Number,
    D: Device,
    A: Activation<'a, T, D, Dim2<SAMPLES, O>>,
{
    pub fn new() -> Self {
        todo!()
    }
    pub fn forward<const N: usize, IS2: MayDim2<N, I>>(
        &self,
        inputs: &Matrix<T, D, IS2>,
    ) -> Matrix<T, D, Dim2<N, O>> {
        todo!()
    }

    pub fn forward_samples<IS>(
        &mut self,
        inputs: &Matrix<'a, T, D, IS>,
    ) -> Matrix<T, D, Dim2<SAMPLES, O>>
    where
        D: CloneBuf<'a, T, IS>
            + CloneBuf<'a, T, Dim2<SAMPLES, O>>
            + Gemm<T, IS, Dim2<I, O>, Dim2<SAMPLES, O>>
            + RowOp<T, Dim2<SAMPLES, O>, Dim2<1, O>, D>
            + ToDim<T, IS, Dim2<SAMPLES, I>>,
        D::Ptr<T, IS>: ShallowCopy,
        D::Ptr<T, Dim2<SAMPLES, O>>: ShallowCopy,
        IS: MayDim2<SAMPLES, I>,
    {
        self.inputs = Some(inputs.shallow_or_clone().to_dims());

        let mut forward = inputs.gemm(&self.weights);

        if let Some(bias) = &self.bias {
            forward.add_row_mut(bias);
        }

        self.activation_inputs = Some(forward.shallow_or_clone());

        // activation function
        A::forward(forward)
    }

    pub fn backward</*GS*/>(&mut self, grad: &Matrix<'a, T, D>) -> Matrix<'a, T, D>
    where 
        //GS: Shape,
        D: TransposeOp<T, Dim2<I, O>>
            + TransposeOp<T, Dim2<SAMPLES, I>>
            + ToDim<T, Dim2<I, O>, ()>
            + ToDim<T, Dim2<1, O>, ()>
            + AdditionalOps<T, Dim2<I, O>>
            + AdditionalOps<T, Dim2<1, O>>
            + AssignOps<T>
            + Gemm<T>
            + RowOp<T>
            + SumOps<T>,
    {
        if self.bias.is_some() {
            self.dbias = Some(grad.sum_rows());
        }

        let mut dweights = self.inputs.as_ref().unwrap().T().gemm(grad);

        if self.l2_reg > T::zero() {
            dweights += (&self.weights * (self.l2_reg * T::two())).to_dims();

            if let Some(dbias) = &mut self.dbias {
                *dbias += (self.bias.as_ref().unwrap() * (self.l2_reg * T::two())).to_dims()
            }
        }
        self.dweights = Some(dweights);

        grad.gemm(&self.weights.T())
    }

    pub fn forward_const_n<const N: usize, IS: MayDim2<N, I>, OS: MayDim2<N, O>>(
        &self,
        inputs: &Matrix<T, D, IS>,
    ) -> Matrix<T, D, OS> {
        todo!()
    }
}

#[test]
fn test_forward_stack() {
    let device = CPU::new();

    let inputs = Matrix::from((&device, 2, 10, [3f32; 2 * 10]));

    let mut lin1 = Linear2::<f32, 10, 100>::new();
    let mut lin2 = Linear2::<f32, 100, 90>::new();

    let x = lin1.forward_samples(&inputs);
    lin2.forward_samples(&x);


    let grad = Matrix::from((&device, 2, 10, [3f32; 2 * 10]));
    lin1.backward(&grad);

    /*let lin1 = Linear::<f32, CPU>::new();
    let lin2 = Linear::<f32, CPU>::new();
    let lin3 = Linear::<f32, CPU>::new();

    let x = lin1.forward(&inputs);
    let x = lin2.forward(&x);*/

    let device = Stack;

    let inputs = Matrix::<_, _, Dim2<1, 10>>::from((&device, 1, 10, [3f32; 1 * 10]));

    let mut lin1 = Linear2::<f32, 10, 100, Stack, ReLU>::new();
    let mut lin2 = Linear2::<f32, 100, 90, Stack, ReLU>::new();

    let out = lin1.forward_samples(&inputs);
    let out = lin2.forward_samples(&out);

    //let lin1 = Linear::<f32, Stack, Dim2<10, 100>, Dim2<1, 100>>::new();
    //let lin2 = Linear::<f32, Stack>::new();
    //let lin3 = Linear::<f32, Stack>::new();

    //let x = lin1.forward(&inputs);
    // let x = lin2.forward(&x);
}
