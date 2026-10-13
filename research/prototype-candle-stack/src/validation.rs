//! CPU-only verification of the prototype's paged-attention readback.

pub fn validate_output(dims: &[usize], expected: [usize; 3], values: &[f32]) -> Result<(), String> {
    if dims != expected {
        return Err(format!("output shape {dims:?}, expected {expected:?}"));
    }
    let expected_count = expected
        .into_iter()
        .try_fold(1usize, usize::checked_mul)
        .ok_or_else(|| "expected output size overflows usize".to_string())?;
    if values.is_empty() || values.len() != expected_count {
        return Err(format!(
            "output contains {} values, expected {expected_count}",
            values.len()
        ));
    }
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!("output[{index}] is not finite: {value}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_output;

    #[test]
    fn rejects_nonfinite_at_previously_unsampled_positions() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut values = vec![0.0; 4096];
            values[1] = value; // The previous stride 257 never visited element 1.
            assert!(validate_output(&[16, 4, 64], [16, 4, 64], &values).is_err());
        }
    }

    #[test]
    fn checks_shape_and_complete_nonempty_readback() {
        assert!(validate_output(&[16, 4, 64], [16, 4, 64], &vec![0.0; 4096]).is_ok());
        assert!(validate_output(&[8, 8, 64], [16, 4, 64], &vec![0.0; 4096]).is_err());
        assert!(validate_output(&[16, 4, 64], [16, 4, 64], &vec![0.0; 4095]).is_err());
        assert!(validate_output(&[0, 4, 64], [0, 4, 64], &[]).is_err());
        assert!(validate_output(&[usize::MAX, 4, 64], [usize::MAX, 4, 64], &[0.0]).is_err());
    }
}
