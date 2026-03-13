use num_complex::Complex;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rustfft::FftPlanner;

/// Build the output dtype: struct {re: f64, im: f64}
fn complex_output(_: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("re".into(), DataType::Float64),
        Field::new("im".into(), DataType::Float64),
    ];
    Ok(Field::new("fft".into(), DataType::Struct(fields)))
}

/// Pack re/im Vec<f64> into a struct series.
fn pack_complex(re: Vec<f64>, im: Vec<f64>, name: &str) -> PolarsResult<Series> {
    let len = re.len();
    let re_series = Float64Chunked::from_vec(PlSmallStr::from("re"), re).into_series();
    let im_series = Float64Chunked::from_vec(PlSmallStr::from("im"), im).into_series();
    let fields = [re_series, im_series];
    StructChunked::from_series(PlSmallStr::from(name), len, fields.iter())
        .map(|ca| ca.into_series())
}

/// Extract re/im from a struct series with fields "re" and "im".
fn unpack_complex(s: &Series) -> PolarsResult<(Vec<f64>, Vec<f64>)> {
    let ca = s.struct_()?;
    let re_s = ca.field_by_name("re")?;
    let im_s = ca.field_by_name("im")?;
    let re_ca = re_s.f64()?;
    let im_ca = im_s.f64()?;
    let re: Vec<f64> = re_ca.into_no_null_iter().collect();
    let im: Vec<f64> = im_ca.into_no_null_iter().collect();
    Ok((re, im))
}

// ── Forward FFT: real f64 input → struct {re, im} ──

#[polars_expr(output_type_func=complex_output)]
fn fft(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].f64()?;
    let n = ca.len();
    if n == 0 {
        return pack_complex(vec![], vec![], "fft");
    }

    let mut buffer: Vec<Complex<f64>> = ca
        .into_no_null_iter()
        .map(|v| Complex::new(v, 0.0))
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    let re: Vec<f64> = buffer.iter().map(|c| c.re).collect();
    let im: Vec<f64> = buffer.iter().map(|c| c.im).collect();
    pack_complex(re, im, "fft")
}

// ── Forward FFT: complex struct input → struct {re, im} ──

#[polars_expr(output_type_func=complex_output)]
fn fft_complex(inputs: &[Series]) -> PolarsResult<Series> {
    let (re_in, im_in) = unpack_complex(&inputs[0])?;
    let n = re_in.len();
    if n == 0 {
        return pack_complex(vec![], vec![], "fft");
    }

    let mut buffer: Vec<Complex<f64>> = re_in
        .into_iter()
        .zip(im_in)
        .map(|(r, i)| Complex::new(r, i))
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    let re: Vec<f64> = buffer.iter().map(|c| c.re).collect();
    let im: Vec<f64> = buffer.iter().map(|c| c.im).collect();
    pack_complex(re, im, "fft")
}

// ── Inverse FFT: struct {re, im} → struct {re, im} (normalized) ──

#[polars_expr(output_type_func=complex_output)]
fn ifft(inputs: &[Series]) -> PolarsResult<Series> {
    let (re_in, im_in) = unpack_complex(&inputs[0])?;
    let n = re_in.len();
    if n == 0 {
        return pack_complex(vec![], vec![], "ifft");
    }

    let mut buffer: Vec<Complex<f64>> = re_in
        .into_iter()
        .zip(im_in)
        .map(|(r, i)| Complex::new(r, i))
        .collect();

    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n);
    ifft.process(&mut buffer);

    // RustFFT does not normalize — divide by N
    let scale = 1.0 / n as f64;
    let re: Vec<f64> = buffer.iter().map(|c| c.re * scale).collect();
    let im: Vec<f64> = buffer.iter().map(|c| c.im * scale).collect();
    pack_complex(re, im, "ifft")
}

// ── Inverse FFT: struct {re, im} → real f64 (takes real part after normalization) ──

#[polars_expr(output_type=Float64)]
fn ifft_real(inputs: &[Series]) -> PolarsResult<Series> {
    let (re_in, im_in) = unpack_complex(&inputs[0])?;
    let n = re_in.len();
    if n == 0 {
        return Ok(Float64Chunked::from_vec(PlSmallStr::from("ifft"), vec![]).into_series());
    }

    let mut buffer: Vec<Complex<f64>> = re_in
        .into_iter()
        .zip(im_in)
        .map(|(r, i)| Complex::new(r, i))
        .collect();

    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n);
    ifft.process(&mut buffer);

    let scale = 1.0 / n as f64;
    let re: Vec<f64> = buffer.iter().map(|c| c.re * scale).collect();
    Ok(Float64Chunked::from_vec(PlSmallStr::from("ifft"), re).into_series())
}

// ── Magnitude: struct {re, im} → f64 ──

#[polars_expr(output_type=Float64)]
fn magnitude(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].struct_()?;
    let re_s = ca.field_by_name("re")?;
    let im_s = ca.field_by_name("im")?;
    let re_ca = re_s.f64()?;
    let im_ca = im_s.f64()?;

    let out: Float64Chunked = re_ca
        .into_iter()
        .zip(im_ca.into_iter())
        .map(|(r, i): (Option<f64>, Option<f64>)| match (r, i) {
            (Some(r), Some(i)) => Some((r * r + i * i).sqrt()),
            _ => None,
        })
        .collect();
    Ok(out.with_name(PlSmallStr::from("magnitude")).into_series())
}

// ── Phase: struct {re, im} → f64 (radians) ──

#[polars_expr(output_type=Float64)]
fn phase(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].struct_()?;
    let re_s = ca.field_by_name("re")?;
    let im_s = ca.field_by_name("im")?;
    let re_ca = re_s.f64()?;
    let im_ca = im_s.f64()?;

    let out: Float64Chunked = re_ca
        .into_iter()
        .zip(im_ca.into_iter())
        .map(|(r, i): (Option<f64>, Option<f64>)| match (r, i) {
            (Some(r), Some(i)) => Some(i.atan2(r)),
            _ => None,
        })
        .collect();
    Ok(out.with_name(PlSmallStr::from("phase")).into_series())
}

// ── Power spectrum: struct {re, im} → f64 (|z|²) ──

#[polars_expr(output_type=Float64)]
fn power_spectrum(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].struct_()?;
    let re_s = ca.field_by_name("re")?;
    let im_s = ca.field_by_name("im")?;
    let re_ca = re_s.f64()?;
    let im_ca = im_s.f64()?;

    let out: Float64Chunked = re_ca
        .into_iter()
        .zip(im_ca.into_iter())
        .map(|(r, i): (Option<f64>, Option<f64>)| match (r, i) {
            (Some(r), Some(i)) => Some(r * r + i * i),
            _ => None,
        })
        .collect();
    Ok(out.with_name(PlSmallStr::from("power")).into_series())
}
