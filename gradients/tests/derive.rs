use std::time::Instant;

use custos_math::{
    nn::{cce, cce_grad},
    Additional,
};
use gradients::{
    correct_classes, Linear, NeuralNetwork, OnehotOp, ReLU, Softmax,
};
use gradients_derive::NeuralNetwork;

use custos::{CLDevice, AsDev, range};
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
    //let device = custos::CPU::new().select();
    let device = CLDevice::get(0).unwrap().select();

    let loader = CSVLoader::new(true);

    let loaded_data = loader
        .load("../../gradients-fallback/datasets/digit-recognizer/train.csv")
        .unwrap();
    //let loaded_data = loader.load("../../../datasets/mnist/mnist_train.csv").unwrap();

    let i = Matrix::from((
        &device,
        (loaded_data.sample_count, loaded_data.features),
        &loaded_data.x,
    ));
    let i = i.divs(255.);

    let y = Matrix::from((&device, (loaded_data.sample_count, 1), &loaded_data.y));
    let y = device.onehot(y);

    let mut net: Network<f32> = Network {
        lin1: Linear::new(784, 128),
        lin2: Linear::new(128, 10),
        lin3: Linear::new(10, 10),
        ..Default::default()
    };

    let mut opt = gradients::Adam::new(0.002);
    //let mut opt = gradients::SGD::new(0.1).momentum(0.8);


    let start = Instant::now();

    for epoch in range(10) {
        let preds = net.forward(i);
        let correct_training = correct_classes(&loaded_data.y.as_usize(), preds) as f32;
        
        let loss = cce(&device, &preds, &y);
        println!(
            "epoch: {epoch}, loss: {loss}, training_acc: {acc}",
            acc = correct_training / loaded_data.sample_count() as f32
        );

        let grad = cce_grad(&device, &preds, &y);
        net.backward(grad);
        opt.step(&device, net.params());
    }

    println!("training duration: {:?}", start.elapsed())
}
