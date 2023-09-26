use custos_math::custos::{range, CPU};
use gradients::{prelude::*, NeuralNetwork};
use purpur::{Apply, Converter, ImageReturn, Transforms};

#[derive(NeuralNetwork)]
struct Network<'a, T, D: Device> {
    lin1: Linear<'a, T, { 100 * 100 * 3 }, 512, D>,
    relu1: ReLU<'a, T, D>,
    lin2: Linear<'a, T, 512, 16, D>,
    relu2: ReLU<'a, T, D>,
    lin3: Linear<'a, T, 16, 3, D>,
    softmax: Softmax<'a, T, D>,
}

#[test]
fn test_berries_net() -> Result<(), std::io::Error> {
    let mut ir = ImageReturn::default();
    let mut trans = Transforms::new(vec![Apply::GetImgRet(&mut ir)]).shuffle();

    trans.apply("../../gradients-fallback/datasets/berries_aug_6xx/train")?;

    let device = CPU::new();

    let x = Matrix::from((
        &device,
        (ir.sample_count(), ir.features()),
        ir.data.as_f32(),
    ));
    let x = x.divs(255.);

    let y = Matrix::from((
        &device,
        (ir.sample_count(), 1),
        ir.get_classes_for_imgs().as_f32(),
    ));
    let y = device.onehot(&y);
    let ru = RandomUniform::new(-0.1, 0.1);
    let mut net = Network {
        lin1: Linear::new(&device, &ru),
        lin2: Linear::new(&device, &ru),
        lin3: Linear::new(&device, &ru),
        ..WithDevice::with(&device)
    };

    let mut opt = Adam::new(1e-4);

    for epoch in range(1000) {
        let predicted = net.forward(&x);

        let (loss, grad) = predicted.cce(&y);
        net.backward(&grad);
        opt.step(&device, net.params());

        println!("epoch: {epoch}, loss: {loss}");
    }
    Ok(())
}
