use std::time::Instant;

use gradients::{network, ReLU, prelude::{Linear, mse}, CPU, create_sine, range};


#[network]
struct SineNet {
    linear1: Linear<1, 64>,
    relu1: ReLU,
    linear2: Linear<64, 64>,
    relu2: ReLU,
    linear3: Linear<64, 1>,
}

#[test]
fn test_net() {
    let device = CPU::new();

    let mut net = SineNet::<f32, _>::with(&device);

    let (x, y) = create_sine(&device, 0, 1000);

    let start = Instant::now();

    for _ in range(1000) {
        let out = net.forward(&x);

        let _loss = (&out - &y).powi(2);

        let grad = (&out - &y) * 2.;
        
        net.backward(&grad);
    }

    println!("elapsed: {:?}", start.elapsed());
}