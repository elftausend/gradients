
use criterion::{criterion_group, criterion_main, Criterion};
use gradients::{Matrix, CPU, Linear, Row, AsDev, set_count};
use purpur::CSVLoader;

fn forward_mut_bias(bench: &mut Criterion) {
    let device = CPU::new().select();
    let loader = CSVLoader::new(true);

    let loaded_data = loader.load("../../gradients-fallback/datasets/digit-recognizer/train.csv").unwrap();
    let i = Matrix::<f32>::from((
        &device,
        (loaded_data.sample_count, loaded_data.features),
        &loaded_data.x,
    ));
    let i = i / 255.;

    let lin = Linear::new(28*28, 256);

    bench.bench_function("forward mut bias", |b| b.iter(|| {
        i.gemm(&lin.weights).add_row_mut(lin.bias);
        set_count(0)
    }));
}

fn forward_bias(bench: &mut Criterion) {
    let device = CPU::new().select();
    let loader = CSVLoader::new(true);

    let loaded_data = loader.load("../../gradients-fallback/datasets/digit-recognizer/train.csv").unwrap();
    let i = Matrix::<f32>::from((
        &device,
        (loaded_data.sample_count, loaded_data.features),
        &loaded_data.x,
    ));
    let i = i / 255.;

    let lin = Linear::new(28*28, 256);
    bench.bench_function("forward bias", |b| b.iter(|| {
        i.gemm(&lin.weights).add_row(lin.bias);
        set_count(0)
    }));
}

fn forward(bench: &mut Criterion) {
    let device = CPU::new().select();
    let loader = CSVLoader::new(true);

    let loaded_data = loader.load("../../gradients-fallback/datasets/digit-recognizer/train.csv").unwrap();
    println!("fin");
    let i = Matrix::<f32>::from((
        &device,
        (loaded_data.sample_count, loaded_data.features),
        &loaded_data.x,
    ));
    let i = i / 255.;

    let mut lin = Linear::new(28*28, 256);
    bench.bench_function("forward", |b| b.iter(|| {
        lin.forward(i);
        set_count(0)
    }));
}

criterion_group!(benches, forward, forward_bias, forward_mut_bias);
criterion_main!(benches);