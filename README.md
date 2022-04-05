# gradients

Deep Learning library using [custos] and [custos-math]

[custos]: https://github.com/elftausend/custos
[custos-math]: https://github.com/elftausend/custos-math

## MNIST [example] 
[example]: https://github.com/elftausend/gradients/examples/mnist.rs
Use a struct to define which layers you want to use:

```rust
use custos_math::{Additional, nn::{cce_grad, cce}};
use gradients::{Linear, ReLU, NeuralNetwork, Softmax, OnehotOp, GetParam, Param, Adam};
use gradients_derive::NeuralNetwork;

use custos::{number::Float, Matrix, opencl::GenericOCL, cpu::TBlas, CLDevice, AsDev, range};
use purpur::CSVLoader;

// Struct for the neural network
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
Load data:
```rust
let device = CLDevice::get(0).unwrap().select();

let loader = CSVLoader::new(true);
let loaded_data = loader.load("").unwrap();

let i = Matrix::from((&device, (loaded_data.sample_count, loaded_data.features), loaded_data.x));
let i = i.divs(255.);

let y = Matrix::from((&device, (loaded_data.sample_count, 1), loaded_data.y));
let y = device.onehot(y);
```