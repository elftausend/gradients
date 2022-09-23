# gradients

[![Crates.io version](https://img.shields.io/crates/v/gradients.svg)](https://crates.io/crates/gradients)
[![Docs](https://docs.rs/gradients/badge.svg?version=0.3.4)](https://docs.rs/gradients/0.3.4/gradients/)

Deep Learning library using [custos] and [custos-math].

external (C) dependencies: OpenCL, CUDA, nvrtc, cublas, a BLAS lib (OpenBLAS, Intel MKL, ...)

[custos]: https://github.com/elftausend/custos
[custos-math]: https://github.com/elftausend/custos-math

## Installation

There are two features available that are enabled by default:
- cuda ... CUDA, nvrtc and cublas must be installed
- opencl ... OpenCL is needed

If you deactivate them (add `default-features = false` and provide no additional features), only the CPU device can be used.

For all feature-configurations, a BLAS library needs to be installed on the system.

```toml
[dependencies]
gradients = "0.3.4"

# to disable the default features (cuda, opencl) and use an own set of features:
#gradients = {version = "0.3.4", default-features = false, features=["opencl"]}
```

## MNIST [example] 

(if this example does not compile, consider looking [here](https://github.com/elftausend/gradients/blob/main/gradients/examples/mnist.rs))

[example]: https://github.com/elftausend/gradients/blob/main/gradients/examples/mnist.rs
Use a struct that implements the NeuralNetwork trait (it is implemented via the `network` attribute) to define which layers you want to use:

```rust
use gradients::prelude::*;

#[network]
pub struct Network {
    lin1: Linear<784, 128>,
    relu1: ReLU,
    lin2: Linear<128, 10>,
    relu2: ReLU,
    lin3: Linear<10, 10>,
    softmax: Softmax,
}
```
Load [data] and create an instance of Network:

You can download the mnist dataset [here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).

[data]: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

```rust
// let device = gradients::CPU::new(); // use cpu (no framework specific features enabled):
// let device = gradients::CudaDevice::new(0)?; // use cuda device (cuda feature enabled):
// use opencl device (opencl feature enabled):
let device = CLDevice::new(0)?;

let mut net = Network::with(&device);

let loader = CSVLoader::new(true);
let loaded_data: CSVReturn<f32> = loader.load("PATH/TO/DATASET/mnist_train.csv")?;

let mut i = Matrix::from((
    &device,
    (loaded_data.sample_count, loaded_data.features),
    &loaded_data.x,
));
i /= 255.;

let y = Matrix::from((&device, (loaded_data.sample_count, 1), &loaded_data.y));
let y = y.onehot();
```

Training loop:

```rust
let mut opt = Adam::new(0.01);

for epoch in range(200) {
    let preds = net.forward(&i);
    let correct_training = correct_classes(&loaded_data.y.as_usize(), &preds) as f32;

    let loss = cce(&device, &preds, &y);
    println!(
        "epoch: {epoch}, loss: {loss}, training_acc: {acc}",
        acc = correct_training / loaded_data.sample_count() as f32
    );

    let grad = cce_grad(&device, &preds, &y);
    net.backward(&grad);
    opt.step(&device, net.params());
}
```

