use custos::CPU;
use gradients::{Conv2D, Matrix};

#[test]
fn test_conv() {
    let device = CPU::new();

    let inputs = Matrix::from((&device, 28, 28, [1.1; 28*28]));

    let _conv = Conv2D::<f32>::new(&device, (28, 28), (3, 3), 5);

    let mut conv = Conv2D::<f32>::new(&device, (28, 28), (3, 3), 5);
    let out = conv.forward(&inputs);
    conv.forward(&out);
}
