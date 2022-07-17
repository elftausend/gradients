use custos::{range, AsDev, CPU};
use custos_math::{nn::mse_grad, Matrix};
use gradients::LinearReg;
use graplot::Scatter;

#[test]
fn test_lg() {
    let device = CPU::new().select();

    let x = [5., 7., 8., 7., 2., 17., 2., 9., 4., 11., 12., 9., 6.];
    let y = [
        99., 86., 87., 88., 111., 86., 70., 87., 94., 78., 77., 85., 86.,
    ];

    let x = Matrix::from((&device, (1, 13), x)).divs(17.);
    let y = Matrix::from((&device, (1, 13), y)).divs(111.);

    let loss_fn = |preds: Matrix<f32>, target: Matrix<f32>| {
        let x = preds - target;
        (x * x).mean()
    };

    //let loss_fn_grad = |preds: Matrix<f32>, target: Matrix<f32>| (preds - target);

    let mut k = 0.;
    let mut d = 0.;

    for _ in range(2000) {
        let y_preds = x.muls(k).adds(d);

        k -= /*(loss_fn_grad(y_preds, y) * x.muls(0.001)).sum()*/ (mse_grad(&device, y_preds, y) * x.muls(0.01)).sum();
        d -= /*(loss_fn_grad(y_preds, y).muls(0.001)).sum()*/ mse_grad(&device, y_preds, y).muls(0.01).sum();
    }

    let y_preds = x.muls(k).adds(d);
    let loss = loss_fn(y_preds, y);
    println!("loss: {loss}, k: {k}, d: {d}")
}

#[test]
fn test_lg_struct() {
    let device = CPU::new().select();

    let x = Matrix::from((
        &device,
        (1, 13),
        [5., 4., 3., 6., 2., 5., 2., 9., 7., 11., 12., 13., 9.],
    ))
    .divs(17.);

    let y = Matrix::from((
        &device,
        (1, 13),
        [
            20., 40., 50., 60., 55., 100., 140., 120., 155., 85., 97., 119., 111.,
        ],
    ))
    .divs(155.);

    let mut lg = LinearReg::new(x, y);

    for _ in range(4000) {
        let loss = lg.step(0.001);
        println!("loss: {loss}");
    }

    let mut scatter = Scatter::new((x.read(), y.read()));
    scatter.add((|x| lg.k as f64 * x + lg.d as f64, "-b"));
    scatter.show()
}
