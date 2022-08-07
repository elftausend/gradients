use gradients::{
    create_sine,
    nn::{mse, mse_grad},
    range, Adam, Linear, NeuralNetwork, ReLU, CPU,
};
use graplot::Plot;

#[derive(NeuralNetwork)]
struct SineNet<'a, T> {
    linear1: Linear<'a, T>,
    relu1: ReLU<'a, T>,
    linear2: Linear<'a, T>,
    relu2: ReLU<'a, T>,
    linear3: Linear<'a, T>,
}

fn main() {
    let device = CPU::new();

    let mut net = SineNet {
        linear1: Linear::new(&device, 1, 64),
        linear2: Linear::new(&device, 64, 32),
        linear3: Linear::new(&device, 32, 1),
        ..Default::default()
    };

    let (x, y) = create_sine(&device, 0, 1000);

    let mut opt = Adam::new(1e-3);

    for epoch in range(2000) {
        let pred = net.forward(&x);
        let loss = mse(&device, &pred, &y);
        let grad = mse_grad(&device, &pred, &y);
        net.backward(&grad);
        opt.step(&device, net.params());

        println!("epoch: {epoch}, loss: {loss}");
    }

    let mut plot = Plot::new((x.read(), y.read()));
    plot.add((x.read(), net.forward(&x).read(), "-r"));
    plot.show()
}
