use custos_math::custos::{CDatatype, prelude::Number};
use custos_math::Matrix;

pub struct LinearReg<'a, T> {
    pub xs: &'a Matrix<'a, T>,
    pub ys: &'a Matrix<'a, T>,
    pub k: T,
    pub d: T,
}

impl<'a, T: CDatatype + Number> LinearReg<'a, T> {
    pub fn new(xs: &'a Matrix<'a, T>, ys: &'a Matrix<'a, T>) -> LinearReg<'a, T> {
        LinearReg {
            xs,
            ys,
            k: T::zero(),
            d: T::zero(),
        }
    }

    pub fn predict(&self, xs: &Matrix<'a, T>) -> Matrix<'a, T> {
        xs * self.k + self.d
    }

    pub fn step(&mut self, lr: T) -> T {
        let y_preds = self.predict(self.xs);

        let loss = y_preds - self.ys;

        self.k -= (&loss * self.xs * (lr * T::two())).sum();
        self.d -= (&loss * (lr * T::two())).sum();

        (&loss * &loss).mean()
    }
}
