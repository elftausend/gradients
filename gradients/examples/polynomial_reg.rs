use gradients::prelude::*;
use graplot::Scatter;

fn main() {
    let device = CPU::new();

    let xs = Matrix::from((
        &device, 1, 26,
        [
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., -1., -2., -3., -4., -5., 
            -6.,-7., -8., -9., -10., -11., -12., -13.,
        ],
    )) / 13.;

    let ys = Matrix::from((
        &device, 1, 26,
        [
            20., 30., 35., 38., 40., 46., 60., 85., 100., 120., 140., 160., 180., 20., 30., 35.,
            38., 40., 46., 60., 85., 100., 120., 140., 160., 180.,
        ],
    )) / 180.;

    let mut lg = LinearReg::new(&xs, &ys);
    let mut poly = PolynomialReg::new(&xs, &ys, 6);

    for _ in range(10000) {
        lg.step(0.001);
        poly.step(0.001);
    }

    let mut scatter = Scatter::new((xs.read(), ys.read()));
    scatter.set_title("linear vs. polynomial regression");
    scatter.add((|x| lg.k as f64 * x + lg.d as f64, "-b"));
    scatter.add((|x: f64| poly.single_predict(x as f32) as f64, "-r"));
    scatter.show();
}
