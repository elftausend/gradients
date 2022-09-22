use gradients::{Linear, CPU};

#[test]
fn test_layer_config() {
    let device = CPU::new();

    let linear = Linear::<f32, 8, 16>::new(&device, ());
    println!("linear: {:?}", linear.weights);    
}