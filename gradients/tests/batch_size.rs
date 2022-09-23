use gradients::{create_sine, prelude::*};

const BATCH_SIZE: usize = 211;

#[network]
struct SineNet {
    linear1: Linear<1, 64>,
    relu1: ReLU,
    linear2: Linear<64, 64>,
    relu2: ReLU,
    linear3: Linear<64, 1>,
}

#[test]
fn test_batch_size() -> custos::Result<()> {
    let device = CLDevice::new(0)?;

    let sine = create_sine(&device, 0, 1000);

    let dataset = Batch::new(&device, BATCH_SIZE, 1000, 1, sine.0.read(), sine.1.read());

    // mind order: there is somewhere a lifetime bug
    let mut net = SineNet::with(&device);

    let mut opt = Adam::new(1e-4);

    for epoch in range(10000) {
        let mut epoch_loss = 0.;

        for (x, y) in &dataset {
            let preds = net.forward(&x);

            epoch_loss += mse(&preds, &y);

            let grad = mse_grad(&preds, &y);
            net.backward(&grad);
            opt.step(&device, net.params());
        }

        println!(
            "epoch: {epoch}, epoch_loss: {}",
            epoch_loss / BATCH_SIZE as f32
        );
    }

    Ok(())
}
