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

#[test]
fn test_l2_reg_loss() {
    let device = CPU::new();

    let l2 = L2Reg::new(1e-5);

    let mut net = SineNet {
        linear1: Linear::new(&device, &l2),
        linear2: Linear::new(&device, &l2),
        linear3: Linear::new(&device, &l2),
        ..WithDevice::with(&device)
    };

    let (x, y) = create_sine(&device, 0, 1000);

    let mut opt = Adam::new(1e-3);

    for epoch in range(1800) {
        let pred = net.forward(&x);
        let loss = mse(&pred, &y);
        let grad = mse_grad(&pred, &y);
        net.backward(&grad);
        opt.step(&device, net.params());

        println!("epoch: {epoch}, reg_loss: {l2}, loss: {loss},");

        println!("weights: {:?}", net.linear1.weights);

        l2.zero();
    }

    let mut plot = Plot::new((x.read(), y.read()));
    plot.add((x.read(), net.forward(&x).read(), "-r"));
    plot.show()
}
