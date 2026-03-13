mod fft;

#[cfg(target_os = "linux")]
use pyo3_polars::PolarsAllocator;

#[cfg(target_os = "linux")]
#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

use num_complex::Complex;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3_polars::PySeries;
use rustfft::FftPlanner;

/// Direct forward FFT bypassing polars plugin dispatch.
/// Takes a polars Series of f64, returns two Series (re, im) as a tuple.
#[pyfunction]
fn fft_direct(series: PySeries) -> PyResult<(PySeries, PySeries)> {
    let s = series.0;
    let ca = s
        .f64()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let n = ca.len();

    let mut buffer: Vec<Complex<f64>> = ca
        .into_no_null_iter()
        .map(|v| Complex::new(v, 0.0))
        .collect();

    let mut planner = FftPlanner::new();
    let fft_plan = planner.plan_fft_forward(n);
    fft_plan.process(&mut buffer);

    let re: Vec<f64> = buffer.iter().map(|c| c.re).collect();
    let im: Vec<f64> = buffer.iter().map(|c| c.im).collect();

    let re_series = Float64Chunked::from_vec("re".into(), re).into_series();
    let im_series = Float64Chunked::from_vec("im".into(), im).into_series();
    Ok((PySeries(re_series), PySeries(im_series)))
}

/// Direct inverse FFT bypassing polars plugin dispatch.
/// Takes two polars Series (re, im) of f64, returns two Series (re, im) as a tuple.
/// Output is normalized by 1/N.
#[pyfunction]
fn ifft_direct(re_series: PySeries, im_series: PySeries) -> PyResult<(PySeries, PySeries)> {
    let re_ca = re_series
        .0
        .f64()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let im_ca = im_series
        .0
        .f64()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let n = re_ca.len();

    let mut buffer: Vec<Complex<f64>> = re_ca
        .into_no_null_iter()
        .zip(im_ca.into_no_null_iter())
        .map(|(r, i)| Complex::new(r, i))
        .collect();

    let mut planner = FftPlanner::new();
    let ifft_plan = planner.plan_fft_inverse(n);
    ifft_plan.process(&mut buffer);

    // RustFFT does not normalize — divide by N
    let scale = 1.0 / n as f64;
    let re: Vec<f64> = buffer.iter().map(|c| c.re * scale).collect();
    let im: Vec<f64> = buffer.iter().map(|c| c.im * scale).collect();

    let re_out = Float64Chunked::from_vec("re".into(), re).into_series();
    let im_out = Float64Chunked::from_vec("im".into(), im).into_series();
    Ok((PySeries(re_out), PySeries(im_out)))
}

#[pymodule]
fn _polars_rfft(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fft_direct, m)?)?;
    m.add_function(wrap_pyfunction!(ifft_direct, m)?)?;
    Ok(())
}
