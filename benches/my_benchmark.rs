use criterion::{criterion_group, criterion_main, Criterion};
use actag_optimized_functions::filters::{median_filter_multithread, adaptive_threshold_multithread, bilateral_filter_multithread};


pub fn criterion_benchmark(c: &mut Criterion) {
    // Create the test vector
    let mut vec_test = Vec::new();
    let row1: Vec<i32> = Vec::from([155, 89, 159, 92, 134, 154, 149, 161, 177, 85]);
	let row2: Vec<i32> = Vec::from([165, 70, 113, 73, 166, 67, 96, 161, 113, 152]);
	let row3: Vec<i32> = Vec::from([77, 176, 112, 80, 155, 85, 130, 106, 187, 78]);
	let row4: Vec<i32> = Vec::from([63, 74, 104, 106, 70, 93, 99, 66, 178, 139]);
	let row5: Vec<i32> = Vec::from([115, 130, 183, 177, 135, 79, 110, 147, 148, 127]);
	let row6: Vec<i32> = Vec::from([175, 65, 134, 92, 125, 158, 128, 98, 139, 75]);
	let row7: Vec<i32> = Vec::from([95, 136, 116, 136, 124, 128, 140, 91, 80, 123]);
	let row8: Vec<i32> = Vec::from([187, 144, 108, 72, 175, 183, 74, 183, 181, 147]);
	let row9: Vec<i32> = Vec::from([78, 80, 108, 100, 66, 123, 190, 176, 101, 99]);
	let row10: Vec<i32> = Vec::from([84, 122, 169, 70, 189, 174, 139, 151, 184, 101]);
	vec_test.push(row1);
    vec_test.push(row2);
    vec_test.push(row3);
    vec_test.push(row4);
    vec_test.push(row5);
    vec_test.push(row6);
    vec_test.push(row7);
    vec_test.push(row8);
    vec_test.push(row9);
    vec_test.push(row10);

    c.bench_function("Median Filter Multithread - R: 15", |b| 
                        b.iter(|| median_filter_multithread(vec_test.to_owned(), 15, None)));
    c.bench_function("Adaptive Threshold Multithread - R: 70, O: 1.0", |b| 
                        b.iter(|| adaptive_threshold_multithread(vec_test.to_owned(), 
                        70, 1.0, None)));
    c.bench_function("Bilaterial Filter Multithread - B: 50, S: 50, R: 70", |b| 
                        b.iter(|| bilateral_filter_multithread(vec_test.to_owned(), 
                        50, 50, 70, None)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);