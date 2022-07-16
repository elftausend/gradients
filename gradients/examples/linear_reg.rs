use custos::{CPU, AsDev, range};
use custos_math::Matrix;
use gradients::LinearReg;
use graplot::Scatter;

fn main() {
    let device = CPU::new().select();

    let x = Matrix::from((&device, (1, 13), [5.,4.,3.,6.,2.,5.,2.,9.,7.,11.,12.,13.,9.]))
        / 17.;

    let y = Matrix::from((&device, (1, 13), [20., 40., 50., 60., 55., 100., 140., 120., 155., 85., 97., 119., 111.,]))
        / 255.;

    let mut lg = LinearReg::new(x, y);
    
    let mut loss_values = vec![0.; 401];

    for i in range(400) {
        loss_values[i] = lg.step(0.001);
    }

    /*let plot = graplot::Plot::new(loss_values);
    plot.show();*/

    let mut scatter = Scatter::new((x.read(), y.read()));
    scatter.add((|x| lg.k as f64 * x + lg.d as f64, "-b"));
    scatter.show();
}