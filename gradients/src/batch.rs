use custos::{cache::CacheReturn, get_count, set_count, Alloc, Cache, RawConv, WriteBuf};
use custos_math::Matrix;

pub struct Batch<'a, T, U, D> {
    x: Vec<T>,
    y: Vec<U>,
    batch_size: usize,
    samples: usize,
    features: usize,
    device: &'a D,
}

impl<'a, T, U, D> Batch<'a, T, U, D> {
    pub fn new(
        device: &'a D,
        batch_size: usize,
        samples: usize,
        features: usize,
        x: Vec<T>,
        y: Vec<U>,
    ) -> Self {
        Batch {
            device,
            batch_size,
            samples,
            features,
            x,
            y,
        }
    }
    #[inline]
    pub fn iter(&'a self) -> Iter<'a, T, U, D>
    where
        D: Alloc<'a, T> + CacheReturn + Alloc<'a, U> + RawConv + WriteBuf<T> + WriteBuf<U>,
        T: Copy + 'a,
        U: Copy + 'a,
    {
        self.into_iter()
    }
}

impl<'a, T, U, D> IntoIterator for &'a Batch<'a, T, U, D>
where
    D: Alloc<'a, T> + CacheReturn + Alloc<'a, U> + RawConv + WriteBuf<T> + WriteBuf<U>,
    T: Copy + 'a,
    U: Copy + 'a,
{
    type Item = (Matrix<'a, T, D>, Matrix<'a, U, D>);

    type IntoIter = Iter<'a, T, U, D>;

    fn into_iter(self) -> Self::IntoIter {
        let remainder = self.samples % self.batch_size;
        let iterations = self.samples / self.batch_size;

        assert!(
            iterations > 0,
            "The batch size cannot be greater than the number of samples."
        );

        Iter {
            x: self.x.as_slice(),
            y: self.y.as_slice(),
            batch_progress: 0,
            batch_size: self.batch_size,
            features: self.features,
            device: self.device,
            remainder,
            iterations,
            current_iter: 0,
            start: get_count(),
        }
    }
}

pub struct Iter<'a, T, U, D> {
    x: &'a [T],
    y: &'a [U],
    batch_size: usize,
    features: usize,
    device: &'a D,
    batch_progress: usize,
    remainder: usize,
    iterations: usize,
    current_iter: usize,
    start: usize,
}

impl<'a, T, U, D> Iterator for Iter<'a, T, U, D>
where
    D: Alloc<'a, T> + CacheReturn + Alloc<'a, U> + RawConv + WriteBuf<T> + WriteBuf<U>,
    T: Copy + 'a,
    U: Copy + 'a,
{
    type Item = (Matrix<'a, T, D>, Matrix<'a, U, D>);

    fn next(&mut self) -> Option<Self::Item> {
        set_count(self.start);
        let mut batch_size = self.batch_size;

        if self.current_iter == self.iterations {
            if self.remainder == 0 {
                return None;
            }
            batch_size = self.remainder;
        }

        if self.current_iter > self.iterations {
            return None;
        }

        self.current_iter += 1;

        let x = &self.x[self.batch_progress..self.batch_progress + batch_size * self.features];

        let mut buf_x = Cache::get(self.device, x.len(), ());
        buf_x.write(x);

        let y = &self.y[self.batch_progress..self.batch_progress + batch_size];

        let mut buf_y = Cache::get::<U, ()>(self.device, y.len(), ());
        buf_y.write(y);

        self.batch_progress += batch_size;

        Some((
            (buf_x, batch_size, self.features).into(),
            (buf_y, batch_size, 1).into(),
        ))
    }
}
