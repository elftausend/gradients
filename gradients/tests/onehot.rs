use custos::{CPU, AsDev, Matrix, CLDevice};
use gradients::OnehotOp;


#[test]
fn test_onehot() {
    let device = CPU::new().select();

    let a = Matrix::from((&device, (4, 1), [1, 2, 0, 3,]));
    device.onehot(a);

    let device = CLDevice::get(0).unwrap().select();

    let a = Matrix::from((&device, (4, 1), [1, 2, 0, 3,]));
    device.onehot(a);
}