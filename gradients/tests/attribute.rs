use std::marker::PhantomData;

use gradients::Linear;
use gradients_derive::network;

struct Linear2<'a, T, const I: usize, const O: usize> {
    p: PhantomData<&'a T>
}

#[network]
struct Net {
    //lin1: Linear<'a, T>,
    lin2: Linear2<784, 10>,
}

struct Net1<'a, T> {
    lin1: Linear<'a, T>,
    lin2: Linear2<'a, T, 784, 10>,
}

#[test]
fn test_attribute_net() {
//    let net = Net {};
}
