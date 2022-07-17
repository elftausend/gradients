use custos::{AsDev, CPU};
use gradients::{Conv2D, Matrix};

#[test]
fn test_conv() {
    let device = CPU::new().select();

    let inputs = Matrix::from((&device, 28, 28, [1.1; 28*28]));

    let mut conv = Conv2D::<f32>::new((28, 28), (3, 3), 5);
    conv.forward(inputs);
}
