use gradients::{CPU, Linear, Matrix, GetParam, NeuralNetwork, number::Float, GenericBlas, CDatatype, CudaTranspose};


pub struct Net<'a, T> {
    lin1: Linear<'a, T>,
    lin2: Linear<'a, T>,
}

impl<'a, T: Float+GenericBlas+CDatatype+CudaTranspose> NeuralNetwork<'a, T> for Net<'a, T> {
    fn forward(&mut self, inputs: &Matrix<'a, T>) -> Matrix<'a, T> {
        let out = self.lin1.forward(inputs);
        let out = self.lin2.forward(&out);
        out
    }

    fn backward(&mut self, grad: &Matrix<'a, T>) -> Matrix<'a, T> {
        let out = self.lin1.backward(grad);
        out
    }

    fn params(&mut self) -> Vec<gradients::Param<T>> {
        todo!()
    }
}

#[test]
fn test_layer() {
    let device = CPU::new();

    let inputs = Matrix::<f32>::from((&device, 5, 16, [0.5; 16*5]));

    let mut lin1 = Linear::<f32>::new(&device, 16, 16);
    let mut lin2 = Linear::<f32>::new(&device, 16, 16);
    let mut lin3 = Linear::<f32>::new(&device, 16, 16);

    let out = lin1.forward(&inputs);
    let out = lin2.forward(&out);


    for _ in 0..100 {
        //let params = lin1.rparams();

        //drop(params);
    }
    
}