use std::time::Instant;

use gradients::{create_sine, prelude::*, NeuralNetwork};

#[derive(NeuralNetwork)]
struct SineNet<'a, T> {
    linear1: Linear<'a, T, 1, 128>,
    relu1: ReLU<'a, T>,
    linear2: Linear<'a, T, 128, 128>,
    relu2: ReLU<'a, T>,
    linear3: Linear<'a, T, 128, 1>,
}

#[test]
fn test_sine_derive() {
    //let device = custos::OpenCL::new(0).unwrap();
    //let device = custos::CudaDevice::new(0).unwrap();
    let device = CPU::new();

    let mut net = SineNet {
        linear1: Linear::new(&device, ()),
        linear2: Linear::new(&device, ()),
        linear3: Linear::new(&device, ()),
        ..Default::default()
    };

    let (x, y) = create_sine(&device, 0, 1000);

    let mut opt = Adam::new(1e-3);

    let start = Instant::now();

    for epoch in range(1000) {
        let pred = net.forward(&x);
        let loss = mse(&pred, &y);
        let grad = mse_grad(&pred, &y);
        net.backward(&grad);
        opt.step(&device, net.params());

        //let traces = device.graph().cache_traces();
        //println!("traces: {traces:?}");
        //device.optimize().unwrap();

        println!("epoch: {epoch}, loss: {loss}");
    }

    println!("elapsed: {:?}", start.elapsed());
}
