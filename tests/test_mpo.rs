use sdkf::mpo::{MpoConfig, MpoSolver};

fn assert_float_rel_eq(a: f64, b: f64, max_rel_diff: f64) {
    let diff = (a - b).abs();
    let max_diff = max_rel_diff * b.abs().max(a.abs());
    assert!(
        diff <= max_diff || diff <= max_rel_diff,
        "{} != {} (rel ε={})",
        a,
        b,
        max_rel_diff
    );
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
fn test_risk_aversion_effect() {
    let mut config = test_config();
    // High risk aversion should prefer less volatile asset
    config.risk_aversion = 10.0;
    let solver = MpoSolver::new(config, &test_covariance()).unwrap();
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
fn test_basic_optimization() {
    let config = test_config();
    let solver = MpoSolver::new(config, &test_covariance()).unwrap();
    let current_weights = vec![0.5, 0.5];

    let weights = solver
        .solve_step(&current_weights, &test_returns())
        .unwrap();

    // Basic checks
    assert_eq!(weights.len(), 2);
    assert_float_rel_eq(weights.iter().sum::<f64>(), 1.0, 1e-6);
    assert!(weights.iter().all(|&w| w >= -1e-6));
}

#[test]
fn test_transaction_cost_effect() {
    let mut config = test_config();
    let solver = MpoSolver::new(config, &test_covariance()).unwrap();

    // High penalty should keep weights close to initial
    config.transaction_cost = 1.0;
    let current_weights = vec![0.8, 0.2];
    let weights = solver
        .solve_step(&current_weights, &test_returns())
        .unwrap();

    assert_float_rel_eq(weights[0], 0.8, 0.05);
    assert_float_rel_eq(weights[1], 0.2, 0.05);
}

#[test]
fn test_penalty_strength_effect() {
    let mut config = test_config();
    let current_weights = vec![0.5, 0.5];

    // 低惩罚时的变化
    config.transaction_cost = 0.01;
    let low_penalty_solver = MpoSolver::new(config, &test_covariance()).unwrap();
    let w_low_penalty = low_penalty_solver
        .solve_step(&current_weights, &test_returns())
        .unwrap();

    // 高惩罚时的变化
    config.transaction_cost = 1.0;
    let high_penalty_solver = MpoSolver::new(config, &test_covariance()).unwrap();
    let w_high_penalty = high_penalty_solver
        .solve_step(&current_weights, &test_returns())
        .unwrap();

    // 高惩罚应导致更小变化
    for (idx, current_weight) in current_weights.iter().enumerate() {
        let w_low = w_low_penalty[idx];
        let w_high = w_high_penalty[idx];
        assert!(
            (w_high - current_weight).abs() < (w_low - current_weight).abs(),
            "low penalty planned weight {} should be greater than high penalty planned weight {}",
            w_low,
            w_high
        );
    }
}

#[test]
fn test_budget_constraints() {
    let config = test_config();
    let solver = MpoSolver::new(config, &test_covariance()).unwrap();
    let current_weights = vec![0.3, 0.7];

    let weights = solver
        .solve_step(&current_weights, &test_returns())
        .unwrap();

    assert_float_rel_eq(weights.iter().sum::<f64>(), 1.0, 1e-6);
}

#[test]
fn test_zero_transaction_cost() {
    let mut config = test_config();
    config.transaction_cost = 0.0;
    let solver = MpoSolver::new(config, &test_covariance()).unwrap();

    let current_weights = vec![0.9, 0.1];
    let weights = solver
        .solve_step(&current_weights, &test_returns())
        .unwrap();

    // Without penalty, current weights shouldn't constrain optimization
    let expected = solver.solve_step(&vec![0.5, 0.5], &test_returns()).unwrap();

    for (weight, expected_weight) in weights.iter().zip(expected.iter()) {
        assert_float_rel_eq(*weight, *expected_weight, 1e-6);
    }
}

#[test]
#[should_panic]
fn test_invalid_input_dimensions() {
    let config = test_config();
    let solver = MpoSolver::new(config, &test_covariance()).unwrap();

    // Wrong current_weights length
    solver.solve_step(&vec![0.5], &test_returns()).unwrap();
}

#[test]
fn test_extreme_returns() {
    let config = test_config();
    let solver = MpoSolver::new(config, &test_covariance()).unwrap();
    let current_weights = vec![0.5, 0.5];

    let extreme_returns = vec![
        vec![0.08, 0.06, 0.04], // Asset A
        vec![0.0, 0.0, 0.0],    // Asset B
    ];

    let weights = solver
        .solve_step(&current_weights, &extreme_returns)
        .unwrap();

    // Should fully allocate to high-return asset
    assert_float_rel_eq(weights[0], 1.0, 1e-6);
    assert_float_rel_eq(weights[1], 0.0, 1e-6);
}

#[test]
fn test_ill_conditioned_covariance() {
    let config = test_config();
    let ill_conditioned_cov = vec![vec![1e-12, 0.0], vec![0.0, 1e-12]];
    let solver = MpoSolver::new(config, &ill_conditioned_cov).unwrap();

    let weights = solver.solve_step(&vec![0.5, 0.5], &test_returns()).unwrap();

    // Should still produce valid weights
    assert_float_rel_eq(weights.iter().sum::<f64>(), 1.0, 1e-6);
}
