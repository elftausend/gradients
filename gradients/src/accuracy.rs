use custos_math::custos::Read;
use custos_math::{Matrix, MaxOps};

pub fn find_idxs<T: Copy + Default + PartialEq, D: Read<T>>(
    search_for: &Matrix<T, D>,
    search_with: &Matrix<T, D>,
) -> Vec<usize> {
    let rows = search_for.rows();
    let search_for = search_for.read_to_vec();
    let search_with = search_with.read_to_vec();
    purpur::utils::find_idxs(rows, &search_for, &search_with)
}

pub fn correct_classes<T: Copy + Default + PartialOrd, D: Read<T> + MaxOps<T, D>>(
    targets: &[usize],
    search_for: &Matrix<T, D>,
) -> usize {
    let search_with = search_for.max_cols();
    let idxs = find_idxs(search_for, &search_with);
    let mut correct = 0;
    for (idx, correct_idx) in idxs.iter().zip(targets) {
        if idx == correct_idx {
            correct += 1;
        }
    }
    correct
}
