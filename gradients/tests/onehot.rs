use custos::{AsDev, CLDevice, Matrix, CPU};
use gradients::OnehotOp;

#[test]
fn test_onehot() {
    let device = CPU::new().select();

    let a = Matrix::from((&device, (4, 1), [1, 2, 0, 3]));
    let onehot = device.onehot(a);
    assert_eq!(onehot.read(), vec![0, 1, 0, 0,
                                    0, 0, 1, 0,
                                    1, 0, 0, 0,
                                    0, 0, 0, 1]);

    let device = CLDevice::get(0).unwrap().select();

    let a = Matrix::from((&device, (4, 1), [1, 2, 0, 3]));
    let onehot = device.onehot(a);
    assert_eq!(onehot.read(), vec![0, 1, 0, 0,
                                    0, 0, 1, 0,
                                    1, 0, 0, 0,
                                    0, 0, 0, 1]);
}
