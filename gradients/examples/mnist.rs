use custos_math::{Additional, nn::{cce_grad, cce}};
use gradients::{Linear, ReLU, NeuralNetwork, Softmax, OnehotOp, GetParam, Param, Adam, correct_classes};
use gradients_derive::NeuralNetwork;

use custos::{number::Float, Matrix, cpu::TBlas, CLDevice, AsDev, range, GenericOCL};
use purpur::{CSVLoader, Converter};


#[derive(NeuralNetwork)]
pub struct Network<T> {
    lin1: Linear<T>,
    relu1: ReLU<T>,
    lin2: Linear<T>,
    relu2: ReLU<T>,
    lin3: Linear<T>,
    softmax: Softmax<T>,
}

fn main() {
    //let device = CPU::new().select();
    let device = CLDevice::get(0).unwrap().select();

    let loader = CSVLoader::new(true);
    let loaded_data = loader.load("PATH/TO/DATASET/mnist_train.csv").unwrap();

    let i = Matrix::from((&device, (loaded_data.sample_count, loaded_data.features), &loaded_data.x));
    let i = i.divs(255.);

    let y = Matrix::from((&device, (loaded_data.sample_count, 1), &loaded_data.y));
    let y = device.onehot(y);

    let mut net = Network {
        lin1: Linear::new(784, 128, 0.1),
        relu1: ReLU::new(),
        lin2: Linear::new(128, 10, 0.1),
        relu2: ReLU::new(),
        lin3: Linear::new(10, 10, 0.1),
        softmax: Softmax::new(),
    };

    let mut opt = Adam::new(0.01);

    for epoch in range(200) {
        let preds = net.forward(i);
        let correct_training = correct_classes( &loaded_data.y.as_usize(), preds) as f32;

        let loss = cce(&device, &preds, &y);
        println!("epoch: {epoch}, loss: {loss}, training_acc: {acc}", acc=correct_training / loaded_data.sample_count() as f32);

        let grad = cce_grad(&device, &preds, &y);
        net.backward(grad);
        opt.step(&device, net.params());
    }
}