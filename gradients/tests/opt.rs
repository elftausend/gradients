use gradients::{create_line, prelude::*};
use graplot::Plot;

#[network]
struct Network {
    lin1: Linear<1, 8>,
    relu1: ReLU,
    lin2: Linear<8, 16>,
    relu2: ReLU,
    lin3: Linear<16, 1>,
}

#[test]
fn test_some_net() {
    let device = CPU::new();

    let mut net = Network::with(&device);

    let (x, y) = create_line::<f32, _>(&device, 0, 800);

    let mut sgd = SGD::<f32>::new(0.05).momentum(0.2);

    for epoch in range(0) {
        let preds = net.forward(&x);
        let loss = mse_loss(&preds, &y);

        println!("epoch: {epoch}, loss: {loss}");

        let grad = mse_grad(&preds, &y);
        net.backward(&grad);
        //sgd.step(&device, net.params());

        let graph = device.graph();
        for node in &graph.nodes {
            let trace = graph.trace_cache_path_raw(node);
            println!("trace: {trace:?}");
        }
        //let cts = device.graph().cache_traces();
        //println!("cts: {cts:?}");
    }

    let mut new_x: Vec<f32> = Vec::with_capacity(1200);
    for add in 0..1200 {
        new_x.push(add as f32 / 1000.);
    }

    let new_x = Matrix::from((&device, 1200, 1, new_x));

    let mut plot = Plot::new((x.read(), y.read()));
    plot.add((new_x.read(), net.forward(&new_x).read(), "-r"));
    //   plot.show();
}
