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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let device = gradients::CPU::new(); // use cpu (no framework specific features enabled):
    // let device = gradients::CudaDevice::new(0)?; // use cuda device (cuda feature enabled):
    // use opencl device (opencl feature enabled):
    //let device = OpenCL::new(0)?;
    let device = gradients::CPU::new();

    let mut net = Network::with(&device);

    let loader = CSVLoader::new(true);
    let loaded_data: CSVReturn<f32> = loader.load("PATH/TO/DATASET/mnist_train.csv")?;

    let mut i = Matrix::from((
        &device,
        (loaded_data.sample_count, loaded_data.features),
        &loaded_data.x,
    ));
    i /= 255.;

    let y = Matrix::from((&device, (loaded_data.sample_count, 1), &loaded_data.y));
    let y = y.onehot();

    let mut opt = Adam::new(0.01);

    for epoch in range(200) {
        let preds = net.forward(&i);
        let correct_training = correct_classes(&loaded_data.y.as_usize(), &preds) as f32;

        let (loss, grad) = preds.cce(&y);
        println!(
            "epoch: {epoch}, loss: {loss}, training_acc: {acc}",
            acc = correct_training / loaded_data.sample_count() as f32
        );

        net.backward(&grad);
        opt.step(&device, net.params());
    }
    Ok(())
}
