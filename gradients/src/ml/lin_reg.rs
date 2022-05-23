use custos::{Matrix, GenericOCL};
use custos_math::{Additional, Sum};

pub struct LinearReg<T> {
    pub xs: Matrix<T>,
    pub ys: Matrix<T>,
    pub k: T,
    pub d: T
}

impl<T: GenericOCL> LinearReg<T> {
    pub fn new(xs: Matrix<T>, ys: Matrix<T>) -> LinearReg<T> {
        LinearReg {
            xs,
            ys,
            k: T::zero(),
            d: T::zero()
        }
    }

    pub fn predict(&self, xs: Matrix<T>) -> Matrix<T> {
        xs.muls(self.k).adds(self.d)
    }

    pub fn step(&mut self, lr: T) -> T {
        let y_preds = self.predict(self.xs);
        
        let loss = y_preds - self.ys;
        
        self.k -= (loss * self.xs.muls(lr)).sum();
        self.d -= loss.muls(lr).sum();

        (loss * loss).mean()
    }
}