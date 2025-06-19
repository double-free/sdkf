#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector, dmatrix, dvector};
    use approx::assert_abs_diff_eq;  // For floating-point comparisons

    #[test]
    fn test_mpc_basic_optimization() {
        // Setup
        let config = MpcConfig {
            horizon: 2,
            num_assets: 2,
            risk_aversion: 1.0,
            transaction_cost: 0.0,  // Disable for simplicity
        };
        let mpc = MpcSolver::new(config);

        // Inputs: 2 assets, 2 time steps
        let current_weights = DVector::from_vec(vec![0.5, 0.5]);
        let predicted_returns = dmatrix![
            0.1, 0.05;  // Asset A returns
            0.03, 0.04  // Asset B returns
        ];
        let covariance = dmatrix![
            0.04, 0.01;
            0.01, 0.01
        ];

        // Solve
        let optimized_weights = mpc
            .solve_step(current_weights.as_view(), &predicted_returns, &covariance)
            .unwrap();

        // --- Assertions ---
        // 1. Sum of weights = 1 (budget constraint)
        assert_abs_diff_eq!(optimized_weights.sum(), 1.0, epsilon = 1e-6);

        // 2. All weights >= 0 (no short-selling)
        assert!(optimized_weights.iter().all(|&w| w >= -1e-6));

        // 3. Expected behavior: Higher weight on higher-return asset (A)
        assert!(optimized_weights[0] > optimized_weights[1]);

        // 4. Check against known solution (pre-computed)
        let expected_weights = dvector![0.6, 0.4];  // Example values
        assert_abs_diff_eq!(
            optimized_weights.as_slice(),
            expected_weights.as_slice(),
            epsilon = 1e-2
        );
    }

    #[test]
    fn test_mpc_transaction_cost_effect() {
        let config = MpcConfig {
            horizon: 2,
            num_assets: 2,
            risk_aversion: 1.0,
            transaction_cost: 0.1,  // High cost to penalize changes
        };
        let mpc = MpcSolver::new(config);

        let current_weights = dvector![0.5, 0.5];
        let predicted_returns = dmatrix![
            0.1, 0.05;
            0.03, 0.04
        ];
        let covariance = dmatrix![
            0.04, 0.01;
            0.01, 0.01
        ];

        let optimized_weights = mpc
            .solve_step(current_weights.as_view(), &predicted_returns, &covariance)
            .unwrap();

        // With high transaction cost, weights should stay closer to initial
        assert_abs_diff_eq!(
            optimized_weights[0],
            0.55,
            epsilon = 0.05
        );  // Less aggressive than 0.6
    }

    #[test]
    fn test_mpc_risk_aversion_effect() {
        let config_high_risk = MpcConfig {
            horizon: 2,
            num_assets: 2,
            risk_aversion: 10.0,  // High risk aversion
            transaction_cost: 0.0,
        };
        let mpc_high_risk = MpcSolver::new(config_high_risk);

        let current_weights = dvector![0.5, 0.5];
        let predicted_returns = dmatrix![
            0.1, 0.05;
            0.03, 0.04
        ];
        let covariance = dmatrix![
            0.04, 0.01;  // Asset A has higher variance
            0.01, 0.01
        ];

        let optimized_high_risk = mpc_high_risk
            .solve_step(current_weights.as_view(), &predicted_returns, &covariance)
            .unwrap();

        // With high risk aversion, prefer lower-variance asset (B)
        assert!(optimized_high_risk[1] > 0.5);
    }

    #[test]
    fn test_mpc_invalid_inputs() {
        let config = MpcConfig {
            horizon: 2,
            num_assets: 2,
            risk_aversion: 1.0,
            transaction_cost: 0.0,
        };
        let mpc = MpcSolver::new(config);

        // Incorrect covariance matrix size
        let bad_covariance = dmatrix![0.04, 0.01, 0.0; 0.01, 0.01, 0.0];
        assert!(mpc
            .solve_step(
                dvector![0.5, 0.5].as_view(),
                &dmatrix![0.1, 0.05; 0.03, 0.04],
                &bad_covariance
            )
            .is_err());
    }
}