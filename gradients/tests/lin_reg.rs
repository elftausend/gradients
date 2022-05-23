use custos::{CPU, AsDev, Matrix, range};
use custos_math::{Additional, Sum, nn::mse_grad};
use gradients::LinearReg;

#[test]
fn test_lg() {
    let device = CPU::new().select();

    let x = [5.,7.,8.,7.,2.,17.,2.,9.,4.,11.,12.,9.,6.];
    let y = [99.,86.,87.,88.,111.,86.,70.,87.,94.,78.,77.,85.,86.];

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

    let x = Matrix::from((&device, (1, 13), [5.,7.,8.,7.,2.,17.,2.,9.,4.,11.,12.,9.,6.]))
        .divs(17.);
    let y = Matrix::from((&device, (1, 13), [99.,86.,87.,88.,111.,86.,70.,87.,94.,78.,77.,85.,86.]))
        .divs(111.);

    let mut lg = LinearReg::new(x, y);
    
    for _ in range(1800) {
        let loss = lg.step(0.001);
        println!("loss: {loss}");
    }

    //let scatter = Scatter::new((&x.read().iter().map(|x| *x as f64).collect::<Vec<f64>>(), &y.read().iter().map(|x| *x as f64).collect::<Vec<f64>>()));
    //let plot = Plot::new(|x| lg.k as f64 * x + lg.d as f64);
    //plot.show();
}