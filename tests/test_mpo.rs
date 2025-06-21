use sdkf::mpo::{MpoConfig, MpoSolver};

fn assert_float_rel_eq(a: f64, b: f64, max_rel_diff: f64) {
    let diff = (a - b).abs();
    let max_diff = max_rel_diff * b.abs().max(a.abs());
    assert!(diff <= max_diff, "{} != {} (rel ε={})", a, b, max_rel_diff);
}

fn test_config() -> MpoConfig {
    MpoConfig {
        horizon: 3,
        num_assets: 2,
        risk_aversion: 1.0,
        transaction_cost: 0.01,
    }
}

fn test_covariance() -> Vec<Vec<f64>> {
    vec![vec![0.04, 0.01], vec![0.01, 0.01]]
}

fn test_returns() -> Vec<Vec<f64>> {
    vec![
        vec![0.08, 0.06, 0.04], // Asset A
        vec![0.03, 0.03, 0.03], // Asset B
    ]
}

#[test]
fn test_basic_optimization() {
    let config = test_config();
    let solver = MpoSolver::new(config, &test_covariance()).unwrap();
    let current_weights = vec![0.5, 0.5];

    let weights = solver
        .solve_step(&current_weights, &test_returns())
        .unwrap();

    // Basic sanity checks
    assert_eq!(weights.len(), 2);
    assert_float_rel_eq(weights.iter().sum::<f64>(), 1.0, 1e-6);
    assert!(weights[0] >= -1e-6 && weights[1] >= -1e-6); // Non-negativity

    // Should prefer higher-return asset
    assert!(weights[0] > weights[1]);
}

#[test]
fn test_transaction_cost_effect() {
    let mut config = test_config();
    let solver = MpoSolver::new(config, &test_covariance()).unwrap();

    // High transaction cost should keep weights closer to initial
    config.transaction_cost = 1.0; // Very high cost
    let current_weights = vec![0.5, 0.5];
    let weights = solver
        .solve_step(&current_weights, &test_returns())
        .unwrap();

    assert_float_rel_eq(weights[0], 0.5, 0.05); // Minimal change
}

#[test]
fn test_risk_aversion_effect() {
    let mut config = test_config();
    let solver = MpoSolver::new(config, &test_covariance()).unwrap();

    // High risk aversion should prefer less volatile asset
    config.risk_aversion = 10.0;
    let current_weights = vec![0.5, 0.5];
    let weights = solver
        .solve_step(&current_weights, &test_returns())
        .unwrap();

    // Asset B has lower variance (0.01 vs 0.04)
    assert!(weights[1] > 0.5);
}

#[test]
fn test_extreme_current_weights() {
    let config = test_config();
    let solver = MpoSolver::new(config, &test_covariance()).unwrap();

    // Test when fully invested in one asset
    let current_weights = vec![1.0, 0.0];
    let weights = solver
        .solve_step(&current_weights, &test_returns())
        .unwrap();

    // Should still produce valid weights
    assert_float_rel_eq(weights.iter().sum::<f64>(), 1.0, 1e-6);
    assert!(weights[0] < 1.0); // Should diversify
}

#[test]
fn test_zero_transaction_cost() {
    let mut config = test_config();
    config.transaction_cost = 0.0;
    let solver = MpoSolver::new(config, &test_covariance()).unwrap();

    let current_weights = vec![0.8, 0.2];
    let weights = solver
        .solve_step(&current_weights, &test_returns())
        .unwrap();

    // Without costs, current weights shouldn't influence optimization
    let expected_weights = solver.solve_step(&vec![0.5, 0.5], &test_returns()).unwrap();

    for (weight, expected_weight) in weights.iter().zip(expected_weights.iter()) {
        assert_float_rel_eq(*weight, *expected_weight, 1e-6);
    }
}

#[test]
fn test_constraint_satisfaction() {
    let config = test_config();
    let solver = MpoSolver::new(config, &test_covariance()).unwrap();
    let current_weights = vec![0.3, 0.7];

    // Verify all constraints are satisfied
    let weights = solver
        .solve_step(&current_weights, &test_returns())
        .unwrap();

    // Budget constraint
    assert_float_rel_eq(weights.iter().sum::<f64>(), 1.0, 1e-6);

    // Non-negativity
    assert!(weights.iter().all(|&w| w >= -1e-6));

    // Transaction cost constraints (when λ > 0)
    if config.transaction_cost > 0.0 {
        let delta = weights[0] - current_weights[0];
        assert!(delta.abs() <= config.transaction_cost * 10.0); // Allow some tolerance
    }
}
