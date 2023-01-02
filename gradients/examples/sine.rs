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

    let mut net = SineNet::with(&device);

    let (x, y) = create_sine(&device, 0, 1000);

    let mut opt = Adam::new(2e-3);

    for epoch in range(1800) {
        let pred = net.forward(&x);
        let loss = mse(&pred, &y);
        let grad = mse_grad(&pred, &y);
        net.backward(&grad);
        opt.step(&device, net.params());

        println!("epoch: {epoch}, loss: {loss}");
    }

    let mut plot = Plot::new((x.read_to_vec(), y.read_to_vec()));
    plot.add((x.read_to_vec(), net.forward(&x).read(), "-r"));
    plot.show()
}
