
use clarabel::algebra::{CscMatrix, FloatT};
use clarabel::solver::{DefaultSolver, DefaultSettings, SolverStatus, SupportedConeT};

pub struct MpcConfig {
    pub horizon: usize,          // T
    pub num_assets: usize,       // n
    pub risk_aversion: f64,      // γ
    pub transaction_cost: f64,   // λ
}

pub struct MpcSolver {
    config: MpcConfig,
    p_matrix: CscMatrix<f64>,    // Precomputed P matrix (risk term)
    base_cones: Vec<SupportedConeT>,
}

impl MpcSolver {
    pub fn new(
        config: MpcConfig,
        covariance: &[Vec<f64>],
    ) -> Result<Self, String> {
        let n = config.num_assets;
        let t = config.horizon;
        
        // 1. Build P matrix (block-diagonal γΣ)
        let mut p_data = Vec::new();
        let mut p_row_indices = Vec::new();
        let mut p_col_ptrs = vec![0];
        
        for step in 0..t {
            for i in 0..n {
                for j in 0..n {
                    p_data.push(config.risk_aversion * covariance[i][j]);
                    p_row_indices.push(step * n + i);
                }
            }
            p_col_ptrs.push(p_data.len());
        }
        let p_matrix = CscMatrix::new(n * t, n * t, p_col_ptrs, p_row_indices, p_data);

        // 2. Base cones (budget + non-negativity)
        let mut base_cones = Vec::new();
        for _ in 0..t {
            base_cones.push(SupportedConeT::ZeroConeT(1));  // sum(w_t) = 1
            base_cones.push(SupportedConeT::NonnegativeConeT(n));  // w_t ≥ 0
        }

        Ok(Self {
            config,
            p_matrix,
            base_cones,
        })
    }

    pub fn solve_step(
        &self,
        current_weights: &[f64],      // Current portfolio weights (n x 1)
        predicted_returns: &[Vec<f64>], // n x T matrix
    ) -> Result<Vec<f64>, String> {
        assert_eq!(current_weights.len(), self.config.num_assets);

        let n = self.config.num_assets;
        let t = self.config.horizon;
        let lambda = self.config.transaction_cost;

        // 1. Build full constraint system with transaction costs
        let (a_matrix, b_vec, cones) = self.build_constraints(current_weights);

        // 2. Build q vector (negative returns + transaction costs)
        let mut q = Vec::with_capacity(n * t);
        for step in 0..t {
            for asset in 0..n {
                q.push(-predicted_returns[asset][step]);
            }
        }

        // 3. Solve
        let settings = DefaultSettings::default();
        let mut solver = DefaultSolver::new(
            &self.p_matrix,
            &q,
            &a_matrix,
            &b_vec,
            &cones,
            settings,
        ).map_err(|e| format!("Solver initialization failed: {:?}", e))?;

        solver.solve();

        match solver.solution.status {
            SolverStatus::Solved => {
                // Return only the first step's weights (x[0..n])
                Ok(solver.solution.x[..n].to_vec())
            },
            status => Err(format!("Solver failed: {:?}", status)),
        }
    }

    fn build_constraints(
        &self,
        current_weights: &[f64],
    ) -> (CscMatrix<f64>, Vec<f64>, Vec<SupportedConeT>) {
        let n = self.config.num_assets;
        let t = self.config.horizon;
        let lambda = self.config.transaction_cost;

        // Base constraints (budget + non-negativity)
        let mut a_data = Vec::new();
        let mut a_row_indices = Vec::new();
        let mut a_col_ptrs = vec![0];
        let mut b_vec = Vec::new();
        let mut cones = self.base_cones.clone();

        // Budget constraints: sum(w_t) = 1
        for step in 0..t {
            for asset in 0..n {
                a_data.push(1.0);
                a_row_indices.push(step);
            }
            b_vec.push(1.0);
            a_col_ptrs.push(a_data.len());
        }

        // Non-negativity: w_t ≥ 0
        for step in 0..t {
            for asset in 0..n {
                a_data.push(-1.0);
                a_row_indices.push(t + step * n + asset);
            }
            b_vec.push(0.0);
            a_col_ptrs.push(a_data.len());
        }

        // Transaction cost constraints if λ > 0
        if lambda > 0.0 {
            for step in 0..t {
                if step == 0 {
                    // For first step: t_1 ≥ |w_1 - w_current|
                    for asset in 0..n {
                        // t_1 ≥ w_1 - w_current → -w_1 + t_1 ≥ -w_current
                        a_data.extend([-1.0, 1.0]);
                        a_row_indices.extend([step * n + asset, n * t + asset]);
                        b_vec.push(-current_weights[asset]);
                    }
                    // t_1 ≥ -(w_1 - w_current) → w_1 + t_1 ≥ w_current
                    for asset in 0..n {
                        a_data.extend([1.0, 1.0]);
                        a_row_indices.extend([step * n + asset, n * t + asset]);
                        b_vec.push(current_weights[asset]);
                    }
                } else {
                    // For subsequent steps: t_k ≥ |w_k - w_{k-1}|
                    for asset in 0..n {
                        // t_k ≥ w_k - w_{k-1} → -w_k + w_{k-1} + t_k ≥ 0
                        a_data.extend([-1.0, 1.0, 1.0]);
                        a_row_indices.extend([
                            step * n + asset,
                            (step - 1) * n + asset,
                            n * t + (step - 1) * n + asset
                        ]);
                        b_vec.push(0.0);
                    }
                    // t_k ≥ -(w_k - w_{k-1}) → w_k - w_{k-1} + t_k ≥ 0
                    for asset in 0..n {
                        a_data.extend([1.0, -1.0, 1.0]);
                        a_row_indices.extend([
                            step * n + asset,
                            (step - 1) * n + asset,
                            n * t + (step - 1) * n + asset
                        ]);
                        b_vec.push(0.0);
                    }
                }
                cones.push(SupportedConeT::NonnegativeConeT(n));
                cones.push(SupportedConeT::NonnegativeConeT(n));
            }
        }

        let a_matrix = CscMatrix::new(
            t + n * t + 2 * n * t,  // Rows: budget + non-neg + transaction
            n * t + if lambda > 0.0 { n * t } else { 0 },  // Columns: w + t
            a_col_ptrs,
            a_row_indices,
            a_data,
        );

        (a_matrix, b_vec, cones)
    }
}