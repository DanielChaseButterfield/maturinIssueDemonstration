// use crate::actag;

// #[pyclass]
// struct AcTagParams {
//     #[pyo3(get, set)]
//     tag_family: String,
//     tag_size: f64,
//     tag_fam_hamming_dist: u64,
//     tag_fam_data_bits: u64,
//     tag_diag: f64,
//     tag_area: f64,
//     tags_in_family: Vec<Vec<i32>>,
// }

// #[pymethods]
// impl AcTagParams {
//     #[new]
//     fn new(tag_family: String, tag_size: f64) -> Self {

//         // Store inputs
//         let actag_params = AcTagParams {
//             tag_family,
//             tag_size,
//             tag_fam_hamming_dist: 0,
//             tag_fam_data_bits: 0,
//             tag_diag: 0.0,
//             tag_area: 0.0,
//             tags_in_family: Vec::new(),
//          };

//         // Verify that the tag family is in the correct input format
//         if !tag_family.contains("AcTag")  || !tag_family.contains("h") {
//             return actag_params.get_parsing_error();
//         }

//         // Calculate other useful parameters based on the input parameters
//         actag_params.tag_fam_hamming_dist = match (match tag_family.split('h').nth(1) {
//             Some(hamming_dist) => hamming_dist,
//             None => return actag_params.get_parsing_error(),
//         }).parse() {
//             Ok(hamming_dist) => hamming_dist,
//             Err(_) => actag_params.get_parsing_error(),
//         };

//         tag_fam_data_bits = match (match (match tag_family.split('h').nth(0) {
//             Some(data_bits) => data_bits,
//             None => return actag_params.get_parsing_error(),
//         }).split("AcTag").nth(1) {
//             Some(data_bits) => data_bits,
//             None => return actag_params.get_parsing_error(),
//         }).parse() {
//             Ok(data_bits) => data_bits,
//             Err(_) => return actag_params.get_parsing_error(),
//         };

//         actag_params.tag_diag = (2.0 * tag_size.powi(2)).sqrt();
//         tag_area = tag_size.powi(2);

//         // Get vectors of all tags in the family
//         tags_in_family = actag_params.get_tags_in_family();

//         // Return the result
//         actag_params
//     }

//     fn get_parsing_error(&self) -> PyResult<String> {
//         Err(PyValueError::new_err("AcTag family must be in the format 'AcTag<#data_bits>h<#hamming_dist>', eg. AcTag24h8".to_string()))
//     }

// }