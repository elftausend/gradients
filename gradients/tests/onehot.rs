use custos::{CLDevice, CPU};
use custos_math::Matrix;
use gradients::OneHotMat;

#[test]
fn test_onehot() {
    let device = CPU::new();

    let a = Matrix::from((&device, (4, 1), [1, 2, 0, 3]));
    let onehot = a.onehot();
    assert_eq!(
        onehot.read(),
        vec![0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1]
    );

    let device = CLDevice::new(0).unwrap();

    let a = Matrix::from((&device, (4, 1), [1, 2, 0, 3]));
    let onehot = a.onehot();
    assert_eq!(
        onehot.read(),
        vec![0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1]
    );
}
