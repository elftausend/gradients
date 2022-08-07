use custos::CDatatype;
use custos_math::Matrix;

pub struct LinearReg<'a, T> {
    pub xs: Matrix<'a, T>,
    pub ys: Matrix<'a, T>,
    pub k: T,
    pub d: T,
}

impl<'a, T: CDatatype> LinearReg<'a, T> {
    pub fn new(xs: Matrix<'a, T>, ys: Matrix<'a, T>) -> LinearReg<'a, T> {
        LinearReg {
            xs,
            ys,
            k: T::zero(),
            d: T::zero(),
        }
    }

    pub fn predict(&self, xs: Matrix<'a, T>) -> Matrix<T> {
        xs * self.k + self.d
    }

    pub fn step(&mut self, lr: T) -> T {
        let y_preds = self.predict(self.xs);

        let loss = y_preds - self.ys;

        self.k -= (loss * self.xs * (lr * T::two())).sum();
        self.d -= (loss * (lr * T::two())).sum();

        (loss * loss).mean()
    }
}
