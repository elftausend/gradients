use std::time::Instant;

use custos_math::nn::{cce, cce_grad, mse, mse_grad};
use gradients::{create_sine, Adam, Linear, NeuralNetwork, ReLU, Softmax, Tanh, OneHotMat};
//use gradients_derive::NeuralNetwork;

use custos::range;
use purpur::{CSVLoader, CSVReturn};

#[derive(NeuralNetwork)]
struct Xor<'a, T> {
    lin1: Linear<'a, T>,
    relu1: Tanh<'a, T>,
    lin2: Linear<'a, T>,
}

#[test]
fn test_xor() -> custos::Result<()> {
    let device = custos::CPU::new();
    //let device = custos::CLDevice::new(0)?;

    let xs = Matrix::from((&device, 4, 2, [0., 0., 0., 1., 1., 0., 1., 1.]));

    let ys = Matrix::from((&device, 4, 2, [1., 0., 0., 1., 0., 1., 1., 0.]));

    let mut net: Xor<f32> = Xor {
        lin1: Linear::new(&device, 2, 4),
        lin2: Linear::new(&device, 4, 2),
        ..Default::default()
    };

    let mut adam = Adam::new(0.001);

    let start = Instant::now();

    for epoch in range(10000) {
        let preds = net.forward(&xs);
        let loss = mse(&device, &preds, &ys);
        println!("epoch: {epoch}, loss: {loss}");

        let grad = mse_grad(&device, &preds, &ys);
        net.backward(&grad);
        adam.step(&device, net.params());
    }

    println!("training duration: {:?}", start.elapsed());
    Ok(())
}

#[test]
fn test_sine() {
    let device = custos::CPU::new();
    //let device = CLDevice::get(0).unwrap().select();

    let (x, y) = create_sine(&device, 0, 1000);
    let mut lin1 = Linear::new(&device, 1, 64);
    let mut relu1 = ReLU::new();
    let mut lin2 = Linear::new(&device, 64, 64);
    let mut relu2 = ReLU::new();
    let mut lin3 = Linear::new(&device, 64, 1);

    for epoch in range(21000) {
        let x = lin1.forward(&x);
        let x = relu1.forward(&x);
        let x = lin2.forward(&x);
        let x = relu2.forward(&x);
        let x = lin3.forward(&x);

        let loss = mse(&device, &x, &y);

        if epoch % 100 == 0 {
            println!("epoch: {epoch}, loss: {loss:?}");
        }

        let grad = mse_grad(&device, &x, &y);

        let x = lin3.backward(&grad);
        let x = relu2.backward(&x);
        let x = lin2.backward(&x);
        let x = relu2.backward(&x);
        lin1.backward(&x);

        lin1.sgd(0.001);
        lin2.sgd(0.001);
        lin3.sgd(0.001);
    }
}

#[test]
fn test_mnist() {
    let device = custos::CPU::new();
    //let device = CLDevice::get(0).unwrap().select();

    let loader = CSVLoader::new(true);

    let loaded_data: CSVReturn<f32> = loader
        .load("../../gradients-fallback/datasets/digit-recognizer/train.csv")
        .unwrap();

    let i = Matrix::from((
        &device,
        (loaded_data.sample_count, loaded_data.features),
        loaded_data.x,
    ));
    let i = i.divs(255.);

    let y = Matrix::from((&device, (loaded_data.sample_count, 1), loaded_data.y));
    let y = y.onehot();

    let mut lin1 = Linear::new(&device, 784, 512);
    let mut relu1 = ReLU::new();
    let mut lin2 = Linear::new(&device, 512, 10);
    let mut relu2 = ReLU::new();
    let mut lin3 = Linear::new(&device, 10, 10);
    let mut softmax = Softmax::new();

    for epoch in range(500) {
        let x = lin1.forward(&i);
        let x = relu1.forward(&x);
        let x = lin2.forward(&x);
        let x = relu2.forward(&x);
        let x = lin3.forward(&x);

        let x = softmax.forward(&x);

        let loss = cce(&device, &x, &y);
        let grad = cce_grad(&device, &x, &y);

        let x = softmax.backward(&grad);

        let x = lin3.backward(&x);
        let x = relu2.backward(&x);
        let x = lin2.backward(&x);
        let x = relu1.backward(&x);
        lin1.backward(&x);

        lin1.sgd(0.1);
        lin2.sgd(0.1);
        lin3.sgd(0.1);

        println!("epoch: {epoch}, loss: {loss}");
    }
}
