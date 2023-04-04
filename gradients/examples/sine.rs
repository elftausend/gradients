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
    //let device = gradients::OpenCL::new(0).unwrap();

    let mut net = SineNet::with(&device);

    let (x, y) = create_sine(&device, 0, 1000);

    // let mut opt = Adam::new(2e-3);
    let mut opt = SGD::new(0.01).momentum(0.);

    let start = std::time::Instant::now();

    for epoch in range(18000) {
        let pred = net.forward(&x);
        let loss = mse_loss(&pred, &y);
        let grad = mse_grad(&pred, &y);
        net.backward(&grad);
        opt.step(&device, net.params());

        println!("epoch: {epoch}, loss: {loss}");
    }

    let mut plot = Plot::new((x.read(), y.read()));
    plot.add((x.read(), net.forward(&x).read(), "-r"));
    plot.show()
}
