use gradients::purpur::{CSVLoader, CSVReturn, Converter};
use gradients::OneHotMat;
use gradients::{
    correct_classes, network,
    nn::{cce, cce_grad},
    range, Adam, CLDevice, Linear, ReLU, Softmax,
};

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
    // use cpu (no features enabled): let device = gradients::CPU::new().select();
    // use cuda device (cuda feature enabled): let device = gradients::CudaDevice::new(0).unwrap().select();
    // use opencl device (opencl feature enabled):
    let device = CLDevice::new(0)?;

    let mut net = Network::with_device(&device);

    let loader = CSVLoader::new(true);
    let loaded_data: CSVReturn<f32> = loader.load("PATH/TO/DATASET/mnist_train.csv")?;

    let i = Matrix::from((
        &device,
        (loaded_data.sample_count, loaded_data.features),
        &loaded_data.x,
    ));
    let i = i / 255.;

    let y = Matrix::from((&device, (loaded_data.sample_count, 1), &loaded_data.y));
    let y = y.onehot();

    let mut opt = Adam::new(0.01);

    for epoch in range(200) {
        let preds = net.forward(&i);
        let correct_training = correct_classes(&loaded_data.y.as_usize(), &preds) as f32;

        let loss = cce(&device, &preds, &y);
        println!(
            "epoch: {epoch}, loss: {loss}, training_acc: {acc}",
            acc = correct_training / loaded_data.sample_count() as f32
        );

        let grad = cce_grad(&device, &preds, &y);
        net.backward(&grad);
        opt.step(&device, net.params());
    }
    Ok(())
}
