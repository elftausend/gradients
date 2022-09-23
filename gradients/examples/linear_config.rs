use gradients::{network, ReLU, Linear, CPU, LinearConfig, RandomUniform};

#[network]
struct SineNet {
    linear1: Linear<1, 64>,
    relu1: ReLU,
    linear2: Linear<64, 64>,
    relu2: ReLU,
    linear3: Linear<64, 1>,
}

fn main() {
    let device = CPU::new();

    let _: SineNet<f32> = SineNet {
        linear1: Linear::new((&device, LinearConfig {
            init: RandomUniform::new(-0.5, 0.5),
            bias: false,
            l2_reg: 2.,
        })),
        ..WithDevice::with(&device)
    };
}

