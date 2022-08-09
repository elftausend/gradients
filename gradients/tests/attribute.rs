use std::time::Instant;

use gradients::{
    correct_classes, network,
    nn::{cce, cce_grad},
    range, Linear, OneHotMat, ReLU, Softmax, CPU,
};
use purpur::{CSVLoader, Converter};

struct _Net1<'a, T> {
    lin1: Linear<'a, T, 784, 10>,
}

#[network]
struct Net {
    lin: Linear<784, 128>,
    relu: ReLU,
    lin1: Linear<128, 10>,
    relu1: ReLU,
    lin2: Linear<10, 10>,
    softmax: Softmax,
}

#[test]
fn test_attribute_net() -> gradients::Result<()> {
    let device = CPU::new();
    //let device = custos::CLDevice::new(0)?;
    let mut net = Net::<f32>::with_device(&device);

    let loader = CSVLoader::new(true);

    let loaded_data =
        loader.load("../../gradients-fallback/datasets/digit-recognizer/train.csv")?;

    let i = Matrix::<f32>::from((
        &device,
        (loaded_data.sample_count, loaded_data.features),
        &loaded_data.x,
    ));
    let i = i / 255.;

    let y = Matrix::from((&device, (loaded_data.sample_count, 1), &loaded_data.y));
    let y = y.onehot();

    let mut opt = gradients::Adam::<f32>::new(0.002);

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
