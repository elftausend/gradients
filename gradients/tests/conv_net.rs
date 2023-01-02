use std::time::Instant;

use gradients::prelude::*;
use gradients::Conv2D;
use gradients::NeuralNetwork;
use purpur::{CSVLoader, Converter};

#[derive(NeuralNetwork)]
pub struct Network<'a, T> {
    conv: Conv2D<'a, T>,
    lin1: Linear<'a, T, { 5 * 26 * 26 }, 128>,
    relu1: ReLU<'a, T>,
    lin2: Linear<'a, T, 128, 10>,
    relu2: ReLU<'a, T>,
    lin3: Linear<'a, T, 10, 10>,
    softmax: Softmax<'a, T>,
}

#[test]
fn test_conv_net() -> custos::Result<()> {
    let device = custos::CPU::new();

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

    let mut net: Network<f32> = Network {
        conv: Conv2D::new(&device, (28, 28), (3, 3), 5),
        lin1: Linear::new(&device, ()),
        lin2: Linear::new(&device, ()),
        lin3: Linear::new(&device, ()),
        ..Default::default()
    };

    let mut opt = gradients::Adam::<f32>::new(0.001);
    //let mut opt = gradients::SGD::new(0.1).momentum(0.8);

    let start = Instant::now();

    /*
    let mut img = 0;
    for epoch in range(100000) {
        if img >= loaded_data.sample_count {
            img = 0;
        }
        let drop = CPU::new();
        let start = img*28*28;
        let single_input = Matrix::from((&drop, 1, 28*28, &i[start..start+28*28]));
        let start = img*10;
        let single_y = Matrix::from((&drop, 1, 10, &y[start..start+10]));


        let preds = net.forward(single_input);
        //let correct_training = correct_classes(&loaded_data.y.as_usize(), preds) as f32;

        let loss = cce(&device, &preds, &single_y);

        if epoch % 100 == 0 {
            println!("epoch: {epoch}, loss: {loss}");
        }

        /*println!(
            "epoch: {epoch}, loss: {loss}, training_acc: {acc}",
            acc = correct_training / loaded_data.sample_count() as f32
        );*/

        img +=1;

        let grad = cce_grad(&device, &preds, &single_y);
        net.backward(grad);
        opt.step(&device, net.params());
    }
    */

    for epoch in range(100000) {
        let preds = net.forward(&i);
        let correct_training = correct_classes(&loaded_data.y.as_usize(), &preds) as f32;

        let loss = cce(&device, &preds, &y);

        //println!("epoch: {epoch}, loss: {loss}");

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
