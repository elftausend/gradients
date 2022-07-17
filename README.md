# gradients

[![Crates.io version](https://img.shields.io/crates/v/gradients.svg)](https://crates.io/crates/gradients)
[![Docs](https://docs.rs/gradients/badge.svg?version=0.1.0)](https://docs.rs/gradients/0.1.0/gradients/)

Deep Learning library using [custos] and [custos-math].

external (C) dependencies: OpenCL, CUDA, nvrtc, cublas, BLAS

[custos]: https://github.com/elftausend/custos
[custos-math]: https://github.com/elftausend/custos-math

## Installation

There are two features available that are enabled by default:
- cuda ... CUDA, nvrtc and cublas are needed to run
- opencl ... OpenCL is needed

If you deactivate them (add `default-features = false` and provide no additional features), only the CPU device can be used.

For all feature configurations, a BLAS library needs to be installed on the system.

```toml
[dependencies]
gradients = "0.1.0"

# to disable the default features (cuda, opencl) and use an own set of features:
#gradients = {version = "0.1.0", default-features = false, features=["opencl"]}
```

## MNIST [example] 

(if this example does not compile, consider looking [here](https://github.com/elftausend/gradients/blob/main/gradients/examples/mnist.rs))

[example]: https://github.com/elftausend/gradients/blob/main/gradients/examples/mnist.rs
Use a struct that implements the NeuralNetwork trait to define which layers you want to use:

```rust
use gradients::purpur::{CSVLoader, CSVReturn, Converter};
use gradients::{
    correct_classes,
    nn::{cce, cce_grad},
    range, Adam, AsDev, CLDevice, Linear, NeuralNetwork, OnehotOp, ReLU, Softmax,
};

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

You can download the mnist dataset [here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).

[data]: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

```rust
// use cpu (no features enabled): let device = gradients::CPU::new().select();
// use cuda device (cuda feature enabled): let device = gradients::CudaDevice::new(0).unwrap().select();
// use opencl device (opencl feature enabled):
let device = CLDevice::new(0).unwrap().select();

let loader = CSVLoader::new(true);
let loaded_data: CSVReturn<f32> = loader.load("PATH/TO/DATASET/mnist_train.csv")?;

let i = Matrix::from((
    &device,
    (loaded_data.sample_count, loaded_data.features),
    &loaded_data.x,
));
let i = i / 255.;

let y = Matrix::from((&device, (loaded_data.sample_count, 1), &loaded_data.y));
let y = device.onehot(y);

let mut net = Network {
    lin1: Linear::new(784, 128),
    lin2: Linear::new(128, 10),
    lin3: Linear::new(10, 10),
    ..Default::default()
};
```

Training loop:

```rust
let mut opt = Adam::new(0.01);

for epoch in range(200) {
    let preds = net.forward(i);
    let correct_training = correct_classes( &loaded_data.y.as_usize(), preds) as f32;

    let loss = cce(&device, &preds, &y);
    println!("epoch: {epoch}, loss: {loss}, training_acc: {acc}", acc=correct_training / loaded_data.sample_count() as f32);

    let grad = cce_grad(&device, &preds, &y);
    net.backward(grad);
    opt.step(&device, net.params());
}
```

