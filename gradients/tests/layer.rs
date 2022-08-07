#![allow(unused)]
//use gradients::{CPU, Linear, Matrix, GetParam, NeuralNetwork, number::Float, GenericBlas, CDatatype, CudaTranspose};

use gradients::{Linear, NeuralNetwork, CPU};


#[derive(NeuralNetwork)]
pub struct Net<'a, T> {
    lin1: Linear<'a, T>,
    lin2: Linear<'a, T>,
}

/*
impl<'a, T: Float+GenericBlas+CDatatype+CudaTranspose> NeuralNetwork<'a, T> for Net<'a, T> {
    fn forward(&mut self, inputs: &Matrix<'a, T>) -> Matrix<'a, T> {
        let out = self.lin1.forward(inputs);
        let out = self.lin2.forward(&out);
        out
    }

    fn backward(&mut self, grad: &Matrix<'a, T>) -> Matrix<'a, T> {
        let out = self.lin1.backward(grad);
        let out = self.lin2.backward(&out);
        out
    }

    fn params(&mut self) -> Vec<gradients::Param<'a, T>> {
        vec![
            self.lin1.params().unwrap(),
            self.lin2.params().unwrap()
        ]
    }
}*/


#[test]
fn test_layer() {
    let device = CPU::new();

    //let mut net = Net::new(&device);

    let mut net = Net {
        lin1: Linear::<f32>::new(&device, 16, 16),
        lin2: Linear::<f32>::new(&device, 16, 16),
        ..Default::default()
    };
    

    let inputs = Matrix::<f32>::from((&device, 5, 16, [0.5; 16*5]));
    let out = net.forward(&inputs);

    println!("out: {out:?}");

    let out = net.backward(&inputs);

    



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