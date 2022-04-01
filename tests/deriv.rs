use custos_math::{Additional, nn::{cce_grad, cce}};
use gradients::{Linear, ReLU, NeuralNetwork, Softmax, OnehotOp};
use gradients_deriv::NeuralNetwork;
use purpur::{number::Float, LoaderBuilder, CSV, CSVLoaderOps, CSVReturn};
use custos::{Matrix, opencl::GenericOCL, cpu::TBlas, CLDevice, AsDev, range};

#[derive(NeuralNetwork)]
pub struct Network<T> {
    lin1: Linear<T>,
    relu1: ReLU<T>,
    lin2: Linear<T>,
    relu2: ReLU<T>,
    lin3: Linear<T>,
    softmax: Softmax<T>,
}

impl Network<f32> {
    pub fn new() -> Network<f32> {
        Network {
            lin1: Linear::new(784, 128, 0.1),
            relu1: ReLU::new(),
            lin2: Linear::new(128, 10, 0.1),
            relu2: ReLU::new(),
            lin3: Linear::new(10, 10, 0.1),
            softmax: Softmax::new(),
        }
    }
}

#[test]
fn test_net() {
    let device = CLDevice::get(0).unwrap().select();

    let loader = LoaderBuilder::<CSV>::new()
        .set_shuffle(true)
        .build();

    let loaded_data = loader.load("../gradients-fallback/datasets/digit-recognizer/train.csv").unwrap();

    let i = Matrix::from((&device, (loaded_data.sample_count, loaded_data.features), loaded_data.x));
    let i = i.divs(255.);

    let y = Matrix::from((&device, (loaded_data.sample_count, 1), loaded_data.y));
    let y = device.onehot(y);

    let mut net = Network::new();

    for epoch in range(200) {
        let preds = net.forward(i);

        let loss = cce(&device, preds, y);
        println!("epoch: {epoch}, loss: {loss}");

        let grad = cce_grad(&device, preds, y);
        net.backward(grad);
    }
}