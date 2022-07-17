use custos::CDatatype;
use custos_math::Matrix;

pub struct LinearReg<T> {
    pub xs: Matrix<T>,
    pub ys: Matrix<T>,
    pub k: T,
    pub d: T,
}

impl<T: CDatatype> LinearReg<T> {
    pub fn new(xs: Matrix<T>, ys: Matrix<T>) -> LinearReg<T> {
        LinearReg {
            xs,
            ys,
            k: T::zero(),
            d: T::zero(),
        }
    }

    pub fn predict(&self, xs: Matrix<T>) -> Matrix<T> {
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
