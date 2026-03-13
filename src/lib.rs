mod fft;

#[cfg(target_os = "linux")]
use pyo3_polars::PolarsAllocator;

#[cfg(target_os = "linux")]
#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pymodule]
fn _polars_rfft(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
