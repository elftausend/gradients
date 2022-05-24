use custos::{Matrix, CPU, AsDev, range};
use custos_math::Additional;
use gradients::{PolynomialReg, LinearReg};
use graplot::Scatter;


#[test]
fn test_poly() {
    let device = CPU::new().select();

    let xs = Matrix::from((&device, (1, 13), [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.,]))
        .divs(13.);

    let ys = Matrix::from((&device, (1, 13), [20., 30., 35., 38., 40., 46., 60., 110., 128., 140., 160., 170., 180.,]))
        .divs(180.);

    
    let mut poly = PolynomialReg::new(xs, ys, 7);

//    let loss = poly.step(0.001);
    let sp = poly.single_predict(4.);
    println!("sp {sp}");


    for _ in range(4000) {
        let loss = poly.step(0.001);
        println!("loss: {loss}");
    }

    //println!("coeffs: {:?}", poly.coeffs);

    let mut lg = LinearReg::new(xs, ys);
    
    for _ in range(4000) {
        lg.step(0.001);
    }

    let mut scatter = Scatter::new((xs.read(), ys.read()));
    scatter.add((|x| lg.k as f64 * x + lg.d as f64, "-b"));
    scatter.add((|x: f64| poly.single_predict(x as f32) as f64, "-r"));
    scatter.show();
    
}
