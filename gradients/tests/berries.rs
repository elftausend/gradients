use custos::{range, CPU, AsDev};
use custos_math::nn::{cce, cce_grad};
use gradients_derive::NeuralNetwork;
use gradients::{NeuralNetwork, Linear, ReLU, Softmax, OnehotOp, Adam};
use purpur::{Transforms, ImageReturn, Apply, Converter};

#[derive(NeuralNetwork)]
struct Network<T> {
    lin1: Linear<T>,
    relu1: ReLU<T>,
    lin2: Linear<T>,
    relu2: ReLU<T>,
    lin3: Linear<T>,
    softmax: Softmax<T>,
}

#[test]
fn test_berries_net() -> Result<(), std::io::Error>{

    let mut ir = ImageReturn::default();
    let mut trans = Transforms::new(vec![
        Apply::GetImgRet(&mut ir),
    ]).shuffle();

    trans.apply("../../gradients-fallback/datasets/berries_aug_6xx/train")?;

    let device = CPU::new().select();

    let x = Matrix::from((&device, (ir.sample_count(), ir.features()), ir.data.as_f32()));
    let x = x.divs(255.);

    let y = Matrix::from((&device, (ir.sample_count(), 1), ir.get_classes_for_imgs().as_f32()));
    let y = device.onehot(y);

    let mut net = Network {
        lin1: Linear::new(100*100*3, 512),
        lin2: Linear::new(512, 16),
        lin3: Linear::new(16, 3),
        ..Default::default()
    };

    let mut opt = Adam::new(1e-4);

    for epoch in range(1000) {
        let predicted = net.forward(x);

        let loss = cce(&device, &predicted, &y);
        let grad = cce_grad(&device, &predicted, &y);
        net.backward(grad);
        opt.step(&device, net.params());
        
        println!("epoch: {epoch}, loss: {loss}");
    }
    Ok(())
}