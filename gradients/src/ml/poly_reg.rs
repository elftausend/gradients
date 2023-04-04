use custos_math::custos::{number::Float, CDatatype, ClearBuf, Device, Alloc};
use custos_math::{AdditionalOps, AssignOps, BaseOps, FnsOps, Matrix, SumOps};

fn single_predict<T: Float>(coeffs: &[T], x: T) -> T {
    let mut sum = T::zero();
    let mut pow = coeffs.len();

    for coeff in coeffs {
        pow -= 1;
        sum += x.powi(pow as i32) * *coeff;
    }
    sum
}

pub struct PolynomialReg<'a, T, D: Device> {
    xs: &'a Matrix<'a, T, D>,
    ys: &'a Matrix<'a, T, D>,
    pub coeffs: Vec<T>,
}

impl<'a, T: CDatatype + Float, D> PolynomialReg<'a, T, D>
where
    D: FnsOps<T> + AdditionalOps<T> + AssignOps<T> + ClearBuf<T>,
{
    pub fn new(xs: &'a Matrix<'a, T, D>, ys: &'a Matrix<'a, T, D>, degree: usize) -> Self {
        PolynomialReg {
            xs,
            ys,
            coeffs: vec![T::one(); degree + 1],
        }
    }

    pub fn single_predict(&self, x: T) -> T {
        single_predict(&self.coeffs, x)
    }

    pub fn predict(&self, xs: &'a Matrix<'a, T, D>) -> Matrix<'a, T, D> 
    where
        D: for<'b> Alloc<'b, T>
    {
        
        let mut sum = Matrix::from((xs.device().retrieve(xs.size(), ()), xs.dims()));
        sum.clear();

        let mut pow = self.coeffs.len();

        for coeff in &self.coeffs {
            pow -= 1;
            sum += xs.powi(pow as i32) * coeff;
        }
        sum
    }

    pub fn step(&mut self, lr: T) -> T
    where
        D: BaseOps<T> + SumOps<T> + for<'b> Alloc<'b, T>,
    {
        let y_preds = self.predict(self.xs);
        let loss = y_preds - self.ys;

        let mut pow = self.coeffs.len();

        for coeff in &mut self.coeffs {
            pow -= 1;
            *coeff -= (&loss * &self.xs.powi(pow as i32) * (lr * T::two())).sum();
        }

        (&loss * &loss).mean()
    }
}

#[cfg(test)]
mod tests {
    use super::single_predict;

    #[test]
    fn test_poly_predict() {
        let coeffs = [5., 2.9, 10.6, 4.3];
        let out = single_predict(&coeffs, 3.);
        assert_eq!(out, 197.2)
    }
}
