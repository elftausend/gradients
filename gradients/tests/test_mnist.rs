use std::time::Instant;

use gradients::prelude::*;

#[network]
pub struct Network {
    lin1: Linear<784, 128>,
    relu1: ReLU,
    lin2: Linear<128, 10>,
    relu2: ReLU,
    lin3: Linear<10, 10>,
    softmax: Softmax,
}

#[test]
fn test_mnist_from_example() -> Result<(), Box<dyn std::error::Error>> {
    // let device = gradients::CPU::new(); // use cpu (no framework specific features enabled):
    // let device = gradients::CudaDevice::new(0)?; // use cuda device (cuda feature enabled):
    // use opencl device (opencl feature enabled):
    let device = gradients::OpenCL::new(1).unwrap();
    // let device = gradients::CPU::new();

    let mut net = Network::with(&device);

    let loader = CSVLoader::new(true);
    let loaded_data: CSVReturn<f32> = loader.load("../../gradients-fallback/datasets/digit-recognizer/train.csv")?;

    let mut i = Matrix::from((
        &device,
        (loaded_data.sample_count, loaded_data.features),
        &loaded_data.x,
    ));
    i /= 255.;

    let y = Matrix::from((&device, (loaded_data.sample_count, 1), &loaded_data.y));
    let y = y.onehot();

    //let mut opt = Adam::new(0.001);
    let mut opt = SGD::new(0.1);

    let start = Instant::now();

    for epoch in range(80) {
        let preds = net.forward(&i);
        let correct_training = correct_classes(&loaded_data.y.as_usize(), &preds) as f32;

        let (loss, grad) = preds.cce(&y);
        println!(
            "epoch: {epoch}, loss: {loss}, training_acc: {acc}",
        //    acc = correct_training / loaded_data.sample_count() as f32
        acc = 3.
        );

        net.backward(&grad);
        opt.step(&device, net.params());
    }

    println!("elapsed: {:?}", start.elapsed());
    Ok(())
}
