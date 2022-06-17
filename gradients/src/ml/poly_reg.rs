use custos::{Matrix, CDatatype, number::Float};
use custos_math::{Additional, Fns, Sum};

pub struct PolynomialReg<T> {
    xs: Matrix<T>,
    ys: Matrix<T>,
    pub coeffs: Vec<T>
}

impl<T: CDatatype+Float> PolynomialReg<T> {
    pub fn new(xs: Matrix<T>, ys: Matrix<T>, degree: usize) -> Self {
        PolynomialReg {
            xs,
            ys,
            coeffs: vec![T::one(); degree +1]
        }
    }

    pub fn single_predict(&self, x: T) -> T {
        let mut sum = T::zero();
        let mut pow = self.coeffs.len();        

        for coeff in &self.coeffs {
            pow -= 1;
            sum += x.powf(T::from_usize(pow)) * *coeff;
            
        }
        sum
    }

    pub fn predict(&self, xs: Matrix<T>) -> Matrix<T> {
        let mut sum = Matrix::from((custos::cached::<T>(xs.size()), xs.dims()));
        sum.clear();
        
        let mut pow = self.coeffs.len();

        for coeff in &self.coeffs {
            pow -= 1;
            sum += &xs.powf(T::from_usize(pow)).muls(*coeff);
        }
        sum
    }

    pub fn step(&mut self, lr: T) -> T {
        let y_preds = self.predict(self.xs);

        let loss = y_preds - self.ys;
        
        let mut pow = self.coeffs.len();

        for coeff in &mut self.coeffs {
            pow -= 1;
            *coeff -= (loss * self.xs.powf(T::from_usize(pow)).muls(lr * T::two())).sum();
        }

        (loss * loss).mean()
    }
}
