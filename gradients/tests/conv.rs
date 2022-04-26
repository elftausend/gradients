use custos::{CPU, AsDev};
use gradients::Conv2D;



#[test]
fn test_conv() {
    let _device = CPU::new().select();
    let conv = Conv2D::<f32>::new((3, 16, 16), 3, 3);
}
