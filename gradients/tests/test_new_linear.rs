use gradients::{
    create_sine,
    number::Number,
    prelude::{Linear2, LinearConfig2, mse, mse_loss},
    AdditionalOps, AssignOps, CDatatype, Device, Params, ReLU2, CPU, range,
};
use graplot::Plot;

pub trait SGDOp<T: Number, D: Device = Self>:
    gradients::BaseOps<T> + AssignOps<T> + AdditionalOps<T>
{
    fn step(&self, lr: T, params: Vec<Params<T, D>>)
    where
        D: gradients::BaseOps<T> + AssignOps<T> + AdditionalOps<T>,
    {
        for mut param in params {
            param.weights -= param.dweights * lr; // sgd.lr

            if let Some(mut bias) = param.bias {
                bias -= param.dbias.expect("Bias is also Some") * lr; // sgd.lr
            }
        }
    }
}

impl<T: Number> SGDOp<T> for CPU {}
impl<T: CDatatype> SGDOp<T> for gradients::OpenCL {}

#[test]
fn test_new_linear_sine2() {
    let device = CPU::new();

    let (inputs, targets) = create_sine(&device, 0, 1000);

    let mut lin1 = Linear2::<f32, 1, 64, _, ReLU2>::new(&device, LinearConfig2::default());
    let mut lin2 = Linear2::<f32, 64, 64, _, ReLU2>::new(&device, LinearConfig2::default());
    let mut lin3 = Linear2::<f32, 64, 1, _>::new(&device, LinearConfig2::default());

    for epoch in range(0..18000) {
        let out = lin1.forward(&inputs);
        let out = lin2.forward(&out);
        let out = lin3.forward(&out);

        let (loss, grad) = mse(&out, &targets);

        println!("epoch: {epoch}, loss: {loss}");

        let out = lin3.backward(grad);
        let out = lin2.backward(out);
        let _out = lin1.backward(out);

        let params = vec![lin3.params(), lin2.params(), lin1.params()];
        device.step(0.01, params);
        
    }

    // let out = lin1.forward(&inputs);
    // let out = lin2.forward(&out);
    // let preds = lin3.forward(&out);
    // 
    // let mut plot = Plot::new((inputs.read(), targets.read()));
    // plot.add((inputs.read(), preds.read(), "-r"));
    // plot.show()
}
