use gradients::prelude::*;

#[test]
fn test_layer_config() {
    let device = CPU::new();

    let linear = Linear::<f32, 8, 16>::new(&device, ());
    println!("linear: {:?}", linear.weights);    
}


#[network]
pub struct Network {
    lin1: Linear<16, 16>,
    relu1: ReLU,
    lin2: Linear<16, 16>,    

}

/*impl<'a, T: Float> WithDevice<'a, T> for Network<'a, T> {
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
}*/

#[test]
fn test_net() {
    let device = CPU::new();

    let net: Network<f32> = Network {
        lin1: Linear::new(&device, LinearConfig {
            init: RandomUniform::new(-5., 5.,),
            ..Default::default()
        }),
        ..WithDevice::with(&device)
    };

    let layer = net.lin2.weights;
    println!("layer: {layer:?}",);
}

