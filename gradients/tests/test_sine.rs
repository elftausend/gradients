use std::time::Instant;

use gradients::{
    create_sine,
    nn::{mse, mse_grad},
    range, Adam, Linear, NeuralNetwork, ReLU, CPU,
};
//use gradients_derive::NeuralNetwork;

#[derive(NeuralNetwork)]
struct SineNet<'a, T> {
    linear1: Linear<'a, T>,
    relu1: ReLU<'a, T>,
    linear2: Linear<'a, T>,
    relu2: ReLU<'a, T>,
    linear3: Linear<'a, T>,
}

#[test]
fn test_sine_derive() {
    //let device = custos::CLDevice::new(0).unwrap().select();
    //let device = custos::CudaDevice::new(0).unwrap().select();
    let device = CPU::new();

    let mut net = SineNet {
        linear1: Linear::new(&device, 1, 128),
        linear2: Linear::new(&device, 128, 128),
        linear3: Linear::new(&device, 128, 1),
        ..Default::default()
    };

    let (x, y) = create_sine(&device, 0, 1000);

    let mut opt = Adam::new(1e-3);

    let start = Instant::now();

    for epoch in range(1000) {
        let pred = net.forward(&x);
        let loss = mse(&device, &pred, &y);
        let grad = mse_grad(&device, &pred, &y);
        net.backward(&grad);
        opt.step(&device, net.params());

        println!("epoch: {epoch}, loss: {loss}");
    }

    println!("elapsed: {:?}", start.elapsed());
}
