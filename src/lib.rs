pub mod sdkf;

// run below tests with: cargo test -- --nocapture
#[cfg(test)]
mod tests {
    use super::sdkf;

    use nalgebra as na;

    #[test]
    fn test_kalman_filter_tracking() {
        // Initial state: position=0, velocity=9.0, actual velocity 10.0
        let initial_state = na::Vector2::new(0.0, 9.0);
        let initial_covariance = na::Matrix2::identity() * 1000.0;
        // sample time
        let sample_time = 0.1;
        let transition_matrix = na::Matrix2::new(1.0, sample_time, 0.0, 1.0);
        let measurement_matrix = na::Matrix1x2::new(1.0, 0.0);
        let process_noise = na::Matrix2::new(1e-5, 0.0, 0.0, 1e-5);
        let measurement_noise = na::Matrix1::new(1.0);

        let mut kf = sdkf::KalmanFilter::new(
            &initial_state,
            &initial_covariance,
            &transition_matrix,
            &measurement_matrix,
            &process_noise,
            &measurement_noise,
        );

        // Simulate measurements at 0.1-second intervals with some noise
        let measurements = vec![1.0, 2.0, 2.9, 4.1, 5.0, 6.1];

        let mut last_pred_err: f32 = 10.0;
        for (i, &measurement) in measurements.iter().enumerate() {
            let observation = na::Matrix1::new(measurement);
            let pred_err = kf.update(&observation).unwrap();

            // Log the state for each step
            println!(
                "Time step {}: State = {:?}, PredErr = {:?}",
                i,
                kf.state(),
                pred_err
            );
            last_pred_err = pred_err.into_scalar();
        }

        // After several steps, the prediction error should be close to 0
        assert!(last_pred_err.abs() < 0.1);
        // with our knowledge, we know the true value is 6.0
        // kalman filter result should be better than observation
        assert!((kf.state()[0] - 6.0).abs() < 0.1);
    }
}
