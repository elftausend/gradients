use gradients_derive::NeuralNetwork;
use gradients::{NeuralNetwork, Linear};

#[derive(NeuralNetwork)]
struct Network<T> {
    lin1: Linear<T>
}