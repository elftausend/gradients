use custos::{range, CPU};
use custos_math::nn::{cce, cce_grad};
use gradients::{Adam, Linear, NeuralNetwork, OnehotOp, ReLU, Softmax};
use purpur::{Apply, Converter, ImageReturn, Transforms};

#[derive(NeuralNetwork)]
struct Network<'a, T> {
    lin1: Linear<'a, T, { 100 * 100 * 3 }, 512>,
    relu1: ReLU<'a, T>,
    lin2: Linear<'a, T, 512, 16>,
    relu2: ReLU<'a, T>,
    lin3: Linear<'a, T, 16, 3>,
    softmax: Softmax<'a, T>,
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

    let mut net = Network {
        lin1: Linear::new(&device),
        lin2: Linear::new(&device),
        lin3: Linear::new(&device),
        ..Default::default()
    };

    let mut opt = Adam::new(1e-4);

    for epoch in range(1000) {
        let predicted = net.forward(&x);

        let loss = cce(&device, &predicted, &y);
        let grad = cce_grad(&device, &predicted, &y);
        net.backward(&grad);
        opt.step(&device, net.params());

        println!("epoch: {epoch}, loss: {loss}");
    }
    Ok(())
}
