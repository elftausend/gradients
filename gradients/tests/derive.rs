use std::time::Instant;

use custos_math::nn::{cce, cce_grad};
use gradients::{correct_classes, Linear, NeuralNetwork, OneHotMat, ReLU, Softmax};

use custos::range;
use purpur::{CSVLoader, Converter};

#[derive(NeuralNetwork)]
pub struct Network<'a, T> {
    lin1: Linear<'a, T, 784, 128>,
    relu1: ReLU<'a, T>,
    lin2: Linear<'a, T, 128, 10>,
    relu2: ReLU<'a, T>,
    lin3: Linear<'a, T, 10, 10>,
    softmax: Softmax<'a, T>,
}

#[test]
fn test_net() -> custos::Result<()> {
    //let device = custos::CPU::new();
    let device = custos::CLDevice::new(0)?;
    //let device = custos::CudaDevice::new(0)?;

    let mut net: Network<f32> = Network {
        lin1: Linear::new(&device),
        lin2: Linear::new(&device),
        lin3: Linear::new(&device),
        ..Default::default()
    };
    let loader = CSVLoader::new(true);

    let loaded_data =
        loader.load("../../gradients-fallback/datasets/digit-recognizer/train.csv")?;
    //let loaded_data = loader.load("../../../datasets/mnist/mnist_train.csv").unwrap();

    let i = Matrix::<f32>::from((
        &device,
        (loaded_data.sample_count, loaded_data.features),
        &loaded_data.x,
    ));
    let i = i / 255.;

    let y = Matrix::from((&device, (loaded_data.sample_count, 1), &loaded_data.y));
    let y = y.onehot();

    let mut opt = gradients::Adam::<f32>::new(0.002);
    //let mut opt = gradients::SGD::new(0.1).momentum(0.5);

    let start = Instant::now();

    for epoch in range(100) {
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

    println!("training duration: {:?}", start.elapsed());
    Ok(())
}
