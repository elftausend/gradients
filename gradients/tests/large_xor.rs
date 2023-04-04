use std::time::Instant;

use gradients::{prelude::*, custos};

#[network]
struct LargeXor {
    lin1: Linear<2, 2048>,
    tanh1: Tanh,
    lin2: Linear<2048, 2048>,
    tanh2: Tanh,
    lin3: Linear<2048, 2>,
    tanh3: Tanh,
}

#[test]
fn test_large_xor() -> custos::Result<()> {
    let device = custos::CPU::new();
    //let device = custos::OpenCL::new(0)?;
    //let device = custos::CudaDevice::new(0)?;

    let xs = Matrix::from((&device, 4, 2, [0., 0., 0., 1., 1., 0., 1., 1.]));

    let ys = Matrix::from((&device, 4, 2, [1., 0., 0., 1., 0., 1., 1., 0.]));

    let mut net: LargeXor<f32, CPU> = LargeXor::with(&device);

    //let mut adam = Adam::new(0.001);

    let mut sgd = gradients::SGD::new(0.1).momentum(0.1);

    let start = Instant::now();

    for epoch in range(500) {
        let preds = net.forward(&xs);
        let loss = mse_loss(&preds, &ys);
        println!("epoch: {epoch}, loss: {loss}");

        let grad = mse_grad(&preds, &ys);
        //let grad = gradients::nn::mse_grad_cl(&device, &preds, &ys);
        net.backward(&grad);
        sgd.step(&device, net.params());
    }

    println!("training duration: {:?}", start.elapsed());
    Ok(())
}
