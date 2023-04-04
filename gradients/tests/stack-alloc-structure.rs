use std::marker::PhantomData;

use gradients::{
    number::Number, prelude::ActivationOps, AdditionalOps, Alloc, AssignOps, CDatatype, CloneBuf,
    Device, Dim2, Gemm, IsShapeIndep, Matrix, MayDim2, PtrConv, RowOp, ShallowCopy, Shape, Stack,
    SumOverOps, ToDim, TransposeOp, CPU,
};

pub trait Activation<'a, T, D: Device, S: Shape = ()> {
    fn forward(inputs: Matrix<'a, T, D, S>) -> Matrix<'a, T, D, S>;
    fn backward(
        inputs: &mut Matrix<'a, T, D, S>,
        grads: Matrix<'a, T, D, S>,
    ) -> Matrix<'a, T, D, S>;
}

impl<'a, T, D: Device + 'a, S: Shape> Activation<'a, T, D, S> for () {
    #[inline]
    fn forward(inputs: Matrix<'a, T, D, S>) -> Matrix<'a, T, D, S> {
        inputs
    }

    #[inline]
    fn backward(
        _inputs: &mut Matrix<'a, T, D, S>,
        grads: Matrix<'a, T, D, S>,
    ) -> Matrix<'a, T, D, S> {
        grads
    }
}

pub struct ReLU;

impl<'a, T, D: ActivationOps<T, S> + AssignOps<T, S>, S: Shape> Activation<'a, T, D, S> for ReLU {
    #[inline]
    fn forward(mut inputs: Matrix<'a, T, D, S>) -> Matrix<'a, T, D, S> {
        inputs.relu_mut();
        inputs
    }

    fn backward(
        inputs: &mut Matrix<'a, T, D, S>,
        mut grads: Matrix<'a, T, D, S>,
    ) -> Matrix<'a, T, D, S> {
        inputs.relu_grad_mut();
        grads *= &*inputs;
        grads
    }
}

#[rustfmt::skip]
pub struct Linear2<'a, T, const I: usize, const O: usize, D = CPU, A = (), const SAMPLES: usize=1> 
where 
    D: Device, 
    //A: Activation<'a, T, D, Dim2<SAMPLES, O>>
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
    // A: Activation<'a, T, D, Dim2<SAMPLES, O>>,
{
    pub fn new(device: &'a D) -> Self
    where
        D: Alloc<'a, T, Dim2<I, O>>,
    {
        Linear2 {
            weights: Matrix::new(device, (I, O)),
            bias: None,
            inputs: None,
            activation_inputs: None,
            dweights: None,
            dbias: None,
            l2_reg: T::default(),
            _p: PhantomData,
        }
    }

    pub fn forward<IS>(
        &mut self,
        inputs: &Matrix<'a, T, D, IS>,
    ) -> Matrix<'a, T, D, Dim2<SAMPLES, O>>
    where
        D: CloneBuf<'a, T, IS>
            + CloneBuf<'a, T, Dim2<SAMPLES, O>>
            + Gemm<T, IS, Dim2<I, O>, Dim2<SAMPLES, O>>
            + RowOp<T, Dim2<SAMPLES, O>, Dim2<1, O>, D>
            + ToDim<T, IS, Dim2<SAMPLES, I>>,
        D::Ptr<T, IS>: ShallowCopy,
        D::Ptr<T, Dim2<SAMPLES, O>>: ShallowCopy,
        IS: MayDim2<SAMPLES, I>,
        A: Activation<'a, T, D, Dim2<SAMPLES, O>>,
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

    pub fn backward<IS>(&mut self, grads: Matrix<'a, T, D, IS>) -> Matrix<'a, T, D>
    where
        D: TransposeOp<T, Dim2<I, O>>
            + PtrConv // TODO: IsShapeIndep
            + TransposeOp<T, Dim2<SAMPLES, I>>
            + ToDim<T, IS, Dim2<SAMPLES, O>>
            + AdditionalOps<T>
            + AssignOps<T>
            + Gemm<T>
            + RowOp<T>
            + SumOverOps<T>,
        A: Activation<'a, T, D>,
        D::Ptr<T, IS>: ShallowCopy,
        IS: Shape,
    {
        let grads = grads.to_dims::<()>();

        // activation fn backwards
        let grads = A::backward(
            self.activation_inputs.as_mut().unwrap().as_dims_mut(),
            grads,
        );

        if self.bias.is_some() {
            self.dbias = Some(grads.sum_rows());
        }

        let mut dweights = self.inputs.as_ref().unwrap().T().gemm(&grads);

        if self.l2_reg > T::zero() {
            dweights += self.weights.as_dims() * (self.l2_reg * T::two());

            if let Some(dbias) = &mut self.dbias {
                *dbias += self.bias.as_ref().unwrap().as_dims() * (self.l2_reg * T::two())
            }
        }
        self.dweights = Some(dweights);

        grads.gemm(&self.weights.T())
    }

    pub fn params<'b>(&'b mut self) -> Params<'b, T, D>
    where
        D: IsShapeIndep,
    {
        let dweights = self
            .dweights
            .as_ref()
            .expect(".backward() on this layer should be called at this moment");

        let weights = self.weights.as_dims_mut();

        let bias = self.bias.as_mut().map(|bias| bias.as_dims_mut());

        Params {
            weights,
            bias,
            dweights,
            dbias: self.dbias.as_ref(),
        }
    }
}

pub struct Params<'a, T, D: Device> {
    weights: &'a mut Matrix<'a, T, D>,
    bias: Option<&'a mut Matrix<'a, T, D>>,
    dweights: &'a Matrix<'a, T, D>,
    dbias: Option<&'a Matrix<'a, T, D>>,
}

struct Network<'a, T, D: Device> {
    lin1: Linear2<'a, T, 10, 100, D, ReLU>,
    lin2: Linear2<'a, T, 100, 10, D, ReLU>,
}

impl<T: Number, D: Device + IsShapeIndep> Network<'_, T, D> {
    pub fn params<'b>(&'b mut self) -> Vec<Params<'b, T, D>> {
        vec![self.lin1.params(), self.lin2.params()]
    }
}

pub trait SGDOp<T: Number, D: Device = Self>:
    gradients::BaseOps<T> + AssignOps<T> + AdditionalOps<T>
{
    fn step(&self, params: Vec<Params<T, D>>)
    where
        D: gradients::BaseOps<T> + AssignOps<T> + AdditionalOps<T>,
    {
        for mut param in params {
            param.weights -= param.dweights * T::one(); // sgd.lr

            if let Some(mut bias) = param.bias {
                bias -= param.dbias.expect("Bias is also Some") * T::one(); // sgd.lr
            }
        }
    }
}

impl<T: Number> SGDOp<T> for CPU {}
impl<T: CDatatype + Number> SGDOp<T> for gradients::OpenCL {}

#[test]
fn test_stack_cpu_network() {
    let device = CPU::new();
    //let device = Stack;

    let mut network: Network<f32, _> = Network {
        lin1: Linear2::new(&device),
        lin2: Linear2::new(&device),
    };

    //let inputs = Matrix::from((&device, 2, 10, [3f32; 2 * 10]));
    let inputs = Matrix::from((&device, 1, 10, [3f32; 1 * 10]));

    for _ in 0..1000 {
        let out = network.lin1.forward(&inputs);
        let out = network.lin2.forward(&out);

        let (loss, grad) = out.cce(&out);

        //let grad = Matrix::from((&device, 2, 10, [3f32; 2 * 10]));
        let out = network.lin2.backward(grad);
        let out = network.lin1.backward(out);

        device.step(network.params());
    }
}

#[test]
fn test_forward_stack() {
    let device = CPU::new();

    let mut lin1 = Linear2::<f32, 10, 100, CPU, ReLU>::new(&device);
    let mut lin2 = Linear2::<f32, 100, 90, CPU, ReLU>::new(&device);

    let inputs = Matrix::from((&device, 2, 10, [3f32; 2 * 10]));

    let x = lin1.forward(&inputs);
    let a = lin2.forward(&x);

    let grad = Matrix::from((&device, 2, 10, [3f32; 2 * 10]));
    let out = lin2.backward(grad);
    let x = lin1.backward(out);

    /*let lin1 = Linear::<f32, CPU>::new();
    let lin2 = Linear::<f32, CPU>::new();
    let lin3 = Linear::<f32, CPU>::new();

    let x = lin1.forward(&inputs);
    let x = lin2.forward(&x);*/

    let device = Stack;

    let inputs = Matrix::<_, _, Dim2<1, 10>>::from((&device, 1, 10, [3f32; 1 * 10]));

    let mut lin1 = Linear2::<f32, 10, 100, Stack, ReLU>::new(&device);
    let mut lin2 = Linear2::<f32, 100, 90, Stack, ReLU>::new(&device);

    let out = lin1.forward(&inputs);
    let out = lin2.forward(&out);

    //let lin1 = Linear::<f32, Stack, Dim2<10, 100>, Dim2<1, 100>>::new();
    //let lin2 = Linear::<f32, Stack>::new();
    //let lin3 = Linear::<f32, Stack>::new();

    //let x = lin1.forward(&inputs);
    // let x = lin2.forward(&x);
}
