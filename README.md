# gradients

Deep Learning library using [custos] and [custos-math].

[custos]: https://github.com/elftausend/custos
[custos-math]: https://github.com/elftausend/custos-math

## MNIST [example] 
[example]: https://github.com/elftausend/gradients/blob/main/gradients/examples/mnist.rs
Use a struct that implements the NeuralNetwork trait to define which layers you want to use:

```rust
use custos_math::{Additional, nn::{cce_grad, cce}};
use gradients::{Linear, ReLU, NeuralNetwork, Softmax, OnehotOp, GetParam, Param, Adam};
use gradients_derive::NeuralNetwork;

use custos::{number::Float, Matrix, opencl::GenericOCL, cpu::TBlas, CLDevice, AsDev, range};
use purpur::CSVLoader;

#[derive(NeuralNetwork)]
pub struct Network<T> {
    lin1: Linear<T>,
    relu1: ReLU<T>,
    lin2: Linear<T>,
    relu2: ReLU<T>,
    lin3: Linear<T>,
    softmax: Softmax<T>,
}
```
Load [data] and create an instance of Network:

[data]: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

```rust
//or use cpu: let device = CPU::new().select();
let device = CLDevice::get(0).unwrap().select();

let loader = CSVLoader::new(true);
let loaded_data = loader.load("../datasets/mnist/_train.csv").unwrap(); //you will need to download the dataset

let i = Matrix::from((&device, (loaded_data.sample_count, loaded_data.features), loaded_data.x));
let i = i.divs(255.);

let y = Matrix::from((&device, (loaded_data.sample_count, 1), loaded_data.y));
let y = device.onehot(y);

let mut net: Network<f32> = Network {
    lin1: Linear::new(784, 128, 0.1),
    relu1: ReLU::new(),
    lin2: Linear::new(128, 10, 0.1),
    relu2: ReLU::new(),
    lin3: Linear::new(10, 10, 0.1),
    softmax: Softmax::new(),
};
```

Training loop:

```rust
let mut opt = Adam::new(0.01);

for epoch in range(200) {
    let preds = net.forward(i);

    let loss = cce(&device, preds, y);
    println!("epoch: {epoch}, loss: {loss}");

    let grad = cce_grad(&device, preds, y);
    net.backward(grad);
    opt.step(&device, net.params());
}
```

