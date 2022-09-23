use gradients::{Linear, CPU, network, NeuralNetwork, WithDevice, number::Float};

#[test]
fn test_layer_config() {
    let device = CPU::new();

    let linear = Linear::<f32, 8, 16>::new(&device);
    println!("linear: {:?}", linear.weights);    
}


#[derive(NeuralNetwork)]
pub struct Network<'a, T> {
    lin1: Linear<'a, T, 16, 16>,
    lin2: Linear<'a, T, 16, 16>,    

}

impl<'a, T: Float> WithDevice<'a, T> for Network<'a, T> {
    fn with_device<'b: 'a, D: gradients::Alloc<T> + gradients::GraphReturn>(device: &'b D) -> Self
    where
        Self: Default,
    {
        Self {
            lin1: Linear::<T, 16, 16>::with_device(device),
            lin2: Linear::<T, 16, 16>::with_device(device),
            ..Default::default()
        }
    }
}

#[test]
fn test_net() {
    let device = CPU::new();

    let net: Network<f32> = Network {
        ..Default::default()
        //..WithDevice::with_device(&device)
    };
}