use gradients::{custos, CPU};
use custos_math::Matrix;
use gradients::find_idxs;

#[test]
fn test_find_idxs_cpu() {
    let device = CPU::new();
    let search_for = Matrix::from((&device, (2, 3), [1, 4, 2, 3, 1, 0]));
    let search_with = Matrix::from((&device, (2, 1), [4, 0]));
    let idxs = find_idxs(&search_for, &search_with);
    assert_eq!(idxs, vec![1, 2])
}

#[cfg(feature = "opencl")]
#[test]
fn test_find_idxs_cl() -> custos::Result<()> {
    let device = custos::OpenCL::new(0)?;
    let search_for = Matrix::from((&device, (2, 3), [1, 4, 2, 3, 1, 0]));
    let search_with = Matrix::from((&device, (2, 1), [4, 0]));
    let idxs = find_idxs(&search_for, &search_with);
    assert_eq!(idxs, vec![1, 2]);
    Ok(())
}
