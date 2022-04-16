use custos_math::{Additional, nn::{cce_grad, cce}};
use gradients::{Linear, ReLU, NeuralNetwork, Softmax, OnehotOp, GetParam, Param, Adam, correct_classes};
use gradients_derive::NeuralNetwork;

use custos::{number::Float, Matrix, GenericOCL, cpu::TBlas, CLDevice, AsDev, range, CPU};
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

#[test]
fn test_net() {
    let _device = CPU::new().select();
    let device = CLDevice::get(0).unwrap().select();

    let loader = CSVLoader::new(true);

    let loaded_data = loader.load("../../gradients-fallback/datasets/digit-recognizer/train.csv").unwrap();
    //let loaded_data = loader.load("../../../datasets/mnist/mnist_train.csv").unwrap();

    let i = Matrix::from((&device, (loaded_data.sample_count, loaded_data.features), &loaded_data.x));
    let i = i.divs(255.);

    let y = Matrix::from((&device, (loaded_data.sample_count, 1), &loaded_data.y));
    let y = device.onehot(y);

    let mut net: Network<f32> = Network {
        lin1: Linear::new(784, 128, 0.1),
        lin2: Linear::new(128, 10, 0.1),
        lin3: Linear::new(10, 10, 0.1),
        ..Default::default()
    };

    let mut opt = Adam::new(0.002);

    for epoch in range(30) {
        let preds = net.forward(i);
        let correct_training = correct_classes( &loaded_data.y.as_usize(), preds) as f32;
        
        let loss = cce(&device, &preds, &y);
        println!("epoch: {epoch}, loss: {loss}, training_acc: {acc}", acc=correct_training / loaded_data.sample_count() as f32);
        
        let grad = cce_grad(&device, &preds, &y);
        net.backward(grad);
        opt.step(&device, net.params());
    }
}