use custos::{Matrix, CPU, AsDev, range, CLDevice};
use custos_math::{Additional, nn::{Softmax, cce}};
use gradients::{onehot, Linear, ReLU, onehot_cl};
use purpur::{LoaderBuilder, CSV, CSVReturn, CSVLoaderOps};



#[test]
fn test_mnist() {
    //let device = CPU::new().select();
    let device = CLDevice::get(0).unwrap().select();

    let loader = LoaderBuilder::<CSV>::new()
        .set_shuffle(true)
        .build();

    let loaded_data: CSVReturn<f32> = loader.load("../gradients-fallback/datasets/digit-recognizer/train.csv").unwrap();

    let i = Matrix::from((&device, (loaded_data.sample_count, loaded_data.features), loaded_data.x));
    let i = i.divs(255.);

    let y = Matrix::from((&device, (loaded_data.sample_count, 1), loaded_data.y));
    let y = onehot_cl(&device, y);

    let mut lin1 = Linear::<f32>::new(784, 512, 0.1);
    let mut relu1 = ReLU::<f32>::new();
    let mut lin2 = Linear::<f32>::new(512, 10, 0.1);
    let mut relu2 = ReLU::<f32>::new();
    let mut lin3 = Linear::<f32>::new(10, 10, 0.1);

    for epoch in range(200) {
        let x = lin1.forward(i);
        let x = relu1.forward(x);
        let x = lin2.forward(x);
        let x = relu2.forward(x);
        let x = lin3.forward(x);

        device.softmax(x);
        let loss = cce(&device, x, y);

        println!("loss: {}", loss);
    }
}