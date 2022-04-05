use custos::{Matrix, AsDev, range, CLDevice};
use custos_math::{Additional, nn::{cce, cce_grad, mse, mse_grad}};
use gradients::{Linear, ReLU, create_sine, OnehotOp, Softmax};
use purpur::{CSVReturn, CSVLoader};


#[test]
fn test_sine() {
    //let device = CPU::new().select();
    let device = CLDevice::get(0).unwrap().select();

    let (x, y) = create_sine(&device, 0, 1000);
    let mut lin1 = Linear::new(1, 64, 1.);
    let mut relu1 = ReLU::new();
    let mut lin2 = Linear::new(64, 64, 1.);
    let mut relu2 = ReLU::new();
    let mut lin3 = Linear::new(64, 1, 1.);

    for epoch in range(21000) {
        let x = lin1.forward(x);
        let x = relu1.forward(x);
        let x = lin2.forward(x);
        let x = relu2.forward(x);
        let x = lin3.forward(x);
        
        let loss = mse(&device, x, y);

        if epoch % 100 == 0 {
            println!("epoch: {epoch}, loss: {loss:?}");
        }
    
        let grad = mse_grad(&device, x, y);

        let x = lin3.backward(grad);
        let x = relu2.backward(x);
        let x = lin2.backward(x);
        let x = relu2.backward(x);
        lin1.backward(x);

        lin1.sgd(0.001);
        lin2.sgd(0.001);
        lin3.sgd(0.001);
    }
}

#[test]
fn test_mnist() {
    //let device = CPU::new().select();
    let device = CLDevice::get(0).unwrap().select();

    let loader = CSVLoader::new(true);

    let loaded_data: CSVReturn<f32> = loader.load("../gradients-fallback/datasets/digit-recognizer/train.csv").unwrap();

    let i = Matrix::from((&device, (loaded_data.sample_count, loaded_data.features), loaded_data.x));
    let i = i.divs(255.);

    let y = Matrix::from((&device, (loaded_data.sample_count, 1), loaded_data.y));
    let y = device.onehot(y);

    let mut lin1 = Linear::new(784, 512, 0.1);
    let mut relu1 = ReLU::new();
    let mut lin2 = Linear::new(512, 10, 0.1);
    let mut relu2 = ReLU::new();
    let mut lin3 = Linear::new(10, 10, 0.1);
    let mut softmax = Softmax::new();

    for epoch in range(500) {
        let x = lin1.forward(i);
        let x = relu1.forward(x);
        let x = lin2.forward(x);
        let x = relu2.forward(x);
        let x = lin3.forward(x);

        let x = softmax.forward(x);
        
        let loss = cce(&device, x, y);
        let grad = cce_grad(&device, x, y);
        
        let x = softmax.backward(grad);

        let x = lin3.backward(x);
        let x = relu2.backward(x);
        let x = lin2.backward(x);
        let x = relu1.backward(x);
        lin1.backward(x);

        lin1.sgd(0.1);
        lin2.sgd(0.1);
        lin3.sgd(0.1);

        println!("epoch: {epoch}, loss: {loss}");
    }
}