use gradients::{
    number::Float, Buffer, CDatatype, CloneBuf, Device, Dim2, Gemm, Matrix, RowOp, ShallowCopy,
    Shape, Stack, CPU, RawConv, PtrType, ToDim
};

pub trait Forward<T, D: Device = CPU, IS: Shape = (), OS: Shape = ()> {
    fn forward(&self, inputs: &Matrix<T, D, IS>) -> Matrix<T, D, OS>;
}

impl<'a, T, D: Device> Forward<T, D> for Linear<'a, T, D> {
    fn forward(&self, inputs: &Matrix<T, D>) -> Matrix<T, D> {
        todo!()
    }
}

impl<'a, T, D: Device, const I: usize, const O: usize, const N: usize>
    Forward<T, D, Dim2<N, I>, Dim2<N, O>> for Linear<'a, T, D, Dim2<I, O>, Dim2<1, O>>
{
    fn forward(&self, inputs: &Matrix<T, D, Dim2<N, I>>) -> Matrix<T, D, Dim2<N, O>> {
        todo!()
    }
}

pub struct Linear2<'a, T, const I: usize, const O: usize, D: Device = CPU, const SAMPLES: usize = 1>
{
    weights: Matrix<'a, T, D, Dim2<I, O>>,
    bias: Option<Matrix<'a, T, D, Dim2<1, O>>>,
    inputs: Option<Matrix<'a, T, D, Dim2<SAMPLES, I>>>,
}

/*pub trait Forward2<const N: usize = 0> {
    fn forward() ->
}*/

impl<'a, T: CDatatype + Float, const I: usize, const O: usize, D: Device, const SAMPLES: usize>
    Linear2<'a, T, I, O, D, SAMPLES>
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

    pub fn forward_samples<IS>(&mut self, inputs: &Matrix<'a, T, D, IS>) -> Matrix<T, D, Dim2<SAMPLES, O>>
    where
        D: CloneBuf<'a, T, IS>
            + Gemm<T, IS, Dim2<I, O>, Dim2<SAMPLES, O>>
            + RowOp<T, Dim2<SAMPLES, O>, Dim2<1, O>, D>
            + ToDim<T, IS, Dim2<SAMPLES, I>>,
        D::Ptr<T, IS>: ShallowCopy,
        IS: MayDim2<SAMPLES, I>,
    {
        self.inputs = Some(inputs.shallow_or_clone().to_dims());
        
        let mut forward = inputs.gemm(&self.weights);

        if let Some(bias) = &self.bias {
            forward.add_row_mut(bias);
        }

        forward
    }

    pub fn forward_const_n<const N: usize, IS: MayDim2<N, I>, OS: MayDim2<N, O>>(
        &self,
        inputs: &Matrix<T, D, IS>,
    ) -> Matrix<T, D, OS> {
        todo!()
    }
}

pub struct Linear<
    'a,
    T,
    /*const I: usize, const O: usize,*/ D: Device = CPU,
    WS: Shape = (),
    BS: Shape = (),
> {
    weights: Matrix<'a, T, D, WS>,
    bias: Matrix<'a, T, D, BS>,
}

impl<'a, T, WS: Shape, BS: Shape, D: Device> Linear<'a, T, D, WS, BS> {
    pub fn new() -> Self {
        todo!()
    }
}

pub struct ReLU {}

pub trait MayDim2<const A: usize, const B: usize>: Shape {}

impl<const A: usize, const B: usize> MayDim2<A, B> for () {}

impl<const A: usize, const B: usize> MayDim2<A, B> for Dim2<A, B> {}

#[test]
fn test_forward_stack() {
    let device = CPU::new();

    let inputs = Matrix::from((&device, 2, 10, [3f32; 2 * 10]));

    let mut lin1 = Linear2::<f32, 10, 100>::new();
    let mut lin2 = Linear2::<f32, 100, 90>::new();

    let x = lin1.forward_samples(&inputs);

    lin2.forward_samples(&x);

    let lin1 = Linear::<f32, CPU>::new();
    let lin2 = Linear::<f32, CPU>::new();
    let lin3 = Linear::<f32, CPU>::new();

    let x = lin1.forward(&inputs);
    let x = lin2.forward(&x);

    let device = Stack;

    let inputs = Matrix::<_, _, Dim2<1, 10>>::from((&device, 1, 10, [3f32; 1 * 10]));

    let mut lin1 = Linear2::<f32, 10, 100, Stack>::new();
    let mut lin2 = Linear2::<f32, 100, 90, Stack>::new();

    let out = lin1.forward_samples(&inputs);
    let out = lin2.forward_samples(&out);

    //let lin1 = Linear::<f32, Stack, Dim2<10, 100>, Dim2<1, 100>>::new();
    //let lin2 = Linear::<f32, Stack>::new();
    //let lin3 = Linear::<f32, Stack>::new();

    //let x = lin1.forward(&inputs);
    // let x = lin2.forward(&x);
}
