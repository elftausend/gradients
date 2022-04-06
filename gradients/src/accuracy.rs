use custos::{Matrix, opencl::GenericOCL};
use custos_math::Max;


pub fn find_idxs<T: Copy+Default+PartialEq>(search_for: Matrix<T>, search_with: Matrix<T>) -> Vec<usize> {
    let rows = search_for.rows();
    let search_for = search_for.read();
    let search_with = search_with.read();
    purpur::utils::find_idxs(rows, &search_for, &search_with)
}

pub fn correct_classes<T: GenericOCL>(targets: &[usize], search_for: Matrix<T>) -> usize {
    let search_with = search_for.max_cols();
    let idxs = find_idxs(search_for, search_with);
    let mut correct = 0;
    for (idx, correct_idx) in idxs.iter().zip(targets) {
        if idx == correct_idx {
            correct += 1;
        }
    }
    correct
}