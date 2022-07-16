use custos::{CPU, AsDev, range};
use custos_math::Matrix;
use gradients::{PolynomialReg, LinearReg};
use graplot::{Scatter, x};

#[test]
fn test_poly_vs_lin_reg() {
    let device = CPU::new().select();

    let xs = Matrix::from((&device, (1, 39), [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.,
                                                           -1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12., -13.,
                                                           -14. - 10., -15. - 10., -16. - 10., -17. - 10., -18. - 10., -19. - 10., -20. - 10., -21. - 10., -22. - 10., -23. - 10., -24. - 10., -25. - 10., -26. - 10.]))
        .divs(13.);

    let ys = Matrix::from((&device, (1, 39), [20., 30., 35., 38., 40., 46., 60., 85., 100., 120., 140., 160., 180.,
                                                           20., 30., 35., 38., 40., 46., 60., 85., 100., 120., 140., 160., 180.,
                                                           -26. + 30., -36. + 30., -41. + 30., -44. + 30., -46. + 30., -52. + 30., -66. + 30., -89. + 30., -106. + 30., -126. + 30., -146. + 30., -166. + 30., -186. + 30.]))
        .divs(180.);

    
    let mut poly = PolynomialReg::new(xs, ys, 5);
    let mut lg = LinearReg::new(xs, ys);
    
    let (mut loss_poly, mut loss_lin) = (0., 0.);
    for _ in range(700000) {
        loss_lin = lg.step(0.001);
        loss_poly = poly.step(0.000001);
        //println!("loss_poly: {loss_poly}")
    }
    
    println!("loss_lin: {loss_lin}, loss_poly: {loss_poly}");

    let mut scatter = Scatter::new((xs.read(), ys.read()));
    scatter.set_title("linear vs. polynomial regression");
    scatter.add((|x| lg.k as f64 * x + lg.d as f64, "-b", x(3.2)));
    scatter.add((|x: f64| poly.single_predict(x as f32) as f64, "-r", x(1.5)));
    scatter.show();
}