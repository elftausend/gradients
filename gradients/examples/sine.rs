use gradients::{
    create_sine,
    nn::{mse, mse_grad},
    range, Adam, AsDev, Linear, NeuralNetwork, ReLU, CPU,
};
use graplot::Plot;

#[derive(NeuralNetwork)]
struct SineNet<T> {
    linear1: Linear<T>,
    relu1: ReLU<T>,
    linear2: Linear<T>,
    relu2: ReLU<T>,
    linear3: Linear<T>,
}

fn main() {
    let device = CPU::new().select();

    let mut net = SineNet {
        linear1: Linear::new(1, 48),
        linear2: Linear::new(48, 32),
        linear3: Linear::new(32, 1),
        ..Default::default()
    };

    let (x, y) = create_sine(&device, 0, 1000);

    let mut opt = Adam::new(1e-3);

    for epoch in range(1600) {
        let pred = net.forward(x);
        let loss = mse(&device, pred, y);
        let grad = mse_grad(&device, pred, y);
        net.backward(grad);
        opt.step(&device, net.params());

        println!("epoch: {epoch}, loss: {loss}");
    }

    let mut plot = Plot::new((x.read(), y.read()));
    plot.add((x.read(), net.forward(x).read(), "-r"));
    plot.show()
}
