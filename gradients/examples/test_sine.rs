use std::time::Instant;

use gradients::{create_sine, prelude::*};

use graplot::Plot;

#[network]
struct SineNet {
    linear1: Linear<1, 64>,
    relu1: ReLU,
    linear2: Linear<64, 64>,
    relu2: ReLU,
    linear3: Linear<64, 1>,
}

fn main() {
    let device = CPU::new();

    /*let mut net = SineNet {
        linear1: Linear::new(&device, LinearConfig {
            init: &Value { val: 1. },
            ..Default::default()
        }),
        linear2: Linear::new(&device, LinearConfig {
            init: &Value { val: 1. },
            ..Default::default()
        }),
        linear3: Linear::new(&device, LinearConfig {
            init: &Value { val: 1. },
            ..Default::default()
        }),
        ..WithDevice::with(&device)
    };*/

    let ru = RandomUniform::new(-0.5, 0.5);
    let mut net = SineNet {
        linear1: Linear::new(&device, LinearConfig {
            init: &ru,
            ..Default::default()
        }),
        linear2: Linear::new(&device, LinearConfig {
            init: &ru,
            ..Default::default()
        }),
        linear3: Linear::new(&device, LinearConfig {
            init: &ru,
            ..Default::default()
        }),
        ..WithDevice::with(&device)
    };

    let (x, y) = create_sine(&device, 0, 1000);

    // let mut opt = Adam::new(2e-3);
    let mut opt = SGD::new(0.0001).momentum(0.);

    let start = Instant::now();

    for epoch in range(1000) {
        let pred = net.forward(&x);

        

        let loss = (&pred - &y).powi(2);

        //let loss = mse_loss(&pred, &y);

        //let grad = mse_grad(&pred, &y);
        let grad = (&pred - &y) * 2.;
        net.backward(&grad);

        //println!("net lin1 dweights: {:?}", net.linear1.dweights);

        opt.step(&device, net.params());

        //println!("epoch: {epoch}, loss: {loss}");
    }

    println!("elapsed: {:?}", start.elapsed());

    let out = net.forward(&x);
    //println!("out: {:?}", out.read());

    let mut plot = Plot::new((x.read(), y.read()));
    plot.add((x.read(), net.forward(&x).read(), "-r"));
    plot.show()
}
