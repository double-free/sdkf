use clarabel::algebra::*;
use clarabel::solver::*;

#[derive(Copy, Clone, Debug)]
pub struct MpoConfig {
    pub horizon: usize,        // T
    pub num_assets: usize,     // n
    pub risk_aversion: f64,    // γ
    pub transaction_cost: f64, // λ
}

pub struct MpoSolver {
    config: MpoConfig,
    p_matrix: CscMatrix<f64>, // Precomputed P matrix (risk term)
    variable_cnt: usize,
}

impl MpoSolver {
    pub fn new(config: MpoConfig, covariance: &[Vec<f64>]) -> Result<Self, String> {
        let mut variable_cnt = config.num_assets;
        if config.transaction_cost > 0.0 {
            // slack variables
            variable_cnt = 2 * variable_cnt;
        }

        let t = config.horizon;

        // 1. Build P matrix (block-diagonal γΣ)
        let mut p_data = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();

        for step in 0..t {
            for i in 0..config.num_assets {
                // row idx: step * n + i
                for j in 0..config.num_assets {
                    // col idx: step * n + j
                    p_data.push(config.risk_aversion * covariance[i][j]);
                    row_indices.push(step * variable_cnt + i);
                    col_indices.push(step * variable_cnt + j);
                }
            }
        }

        let p_matrix = CscMatrix::new_from_triplets(
            variable_cnt * t,
            variable_cnt * t,
            row_indices,
            col_indices,
            p_data,
        );

        println!(
            "Multi-horizon covariance matrix in csc format: {:?}",
            p_matrix
        );
        // correctness check
        let csc_mat_check = p_matrix.check_format();
        if csc_mat_check.is_ok() {
            return Ok(Self {
                config,
                p_matrix,
                variable_cnt,
            });
        }
        return Err(format!("illegal covariance matrix: {:?}", csc_mat_check));
    }

    pub fn solve_step(
        &self,
        current_weights: &[f64],        // Current portfolio weights (n x 1)
        predicted_returns: &[Vec<f64>], // n x T matrix
    ) -> Result<Vec<f64>, String> {
        assert_eq!(current_weights.len(), self.config.num_assets);

        let t = self.config.horizon;

        // 1. Build full constraint system with transaction costs
        let (a_matrix, b_vec, cones) = self.build_constraints(current_weights);

        // 2. Build q vector (negative returns + transaction costs)
        let mut q = vec![0.0; self.variable_cnt * t];
        for step in 0..t {
            for asset in 0..self.config.num_assets {
                q[step * self.variable_cnt + asset] = -predicted_returns[asset][step];
                if self.config.transaction_cost > 0.0 {
                    q[step * self.variable_cnt + self.config.num_assets + asset] =
                        self.config.transaction_cost;
                }
            }
        }

        // 3. Solve
        let settings = DefaultSettings::default();
        let mut solver =
            DefaultSolver::new(&self.p_matrix, &q, &a_matrix, &b_vec, &cones, settings)
                .map_err(|e| format!("Solver initialization failed: {:?}", e))?;

        solver.solve();

        match solver.solution.status {
            SolverStatus::Solved => {
                // Return only the first step's weights (x[0..n])
                Ok(solver.solution.x[..self.config.num_assets].to_vec())
            }
            status => Err(format!("Solver failed: {:?}", status)),
        }
    }

    fn build_constraints(
        &self,
        current_weights: &[f64],
    ) -> (CscMatrix<f64>, Vec<f64>, Vec<SupportedConeT<f64>>) {
        let n = self.config.num_assets;
        let t = self.config.horizon;
        let lambda = self.config.transaction_cost;

        // Base constraints (budget + non-negativity)
        let mut a_data = Vec::new();
        let mut a_row_indices = Vec::new();
        let mut a_col_indices = Vec::new();
        let mut b_vec = Vec::new();
        let mut cones = Vec::new();

        // Budget constraints: sum(w_t) = 1
        for step in 0..t {
            for asset in 0..n {
                // for each step, we have sum(w_t) = 1 so in total t*n 1s.
                a_data.push(1.0);
                a_row_indices.push(b_vec.len());
                a_col_indices.push(step * self.variable_cnt + asset);
            }
            b_vec.push(1.0);
            cones.push(SupportedConeT::ZeroConeT(1)); // sum(w_t) = 1
        }

        // Non-negativity: w_t ≥ 0
        for step in 0..t {
            for asset in 0..n {
                // for each step, each weight >= 0, so in total t*n 1s
                // but row number is different with above constraint.
                a_data.push(-1.0);
                a_row_indices.push(b_vec.len());
                a_col_indices.push(step * self.variable_cnt + asset);
                b_vec.push(0.0);
            }
            cones.push(SupportedConeT::NonnegativeConeT(n)); // w_t ≥ 0
        }

        // Transaction cost constraints if λ > 0
        // add one auxiliary variable t_1, converting t_1 ≥ |w_1 - w_current| to
        // (w_1 - w_current <= t_1), and (w1_ - w_current >= - t_1)
        // t_1 is the extra asset at column n
        if lambda > 0.0 {
            for step in 0..t {
                for asset in 0..n {
                    let w_col = step * self.variable_cnt + asset;
                    // first t*n is for w.
                    let t_col = step * self.variable_cnt + n + asset;
                    if step == 0 {
                        // For first step: t_1 ≥ |w_1 - w_current|

                        // t_1 ≥ w_1 - w_current → -w_1 + t_1 ≥ -w_current
                        a_data.extend([1.0, -1.0]);
                        a_row_indices.extend([b_vec.len(), b_vec.len()]);
                        a_col_indices.extend([w_col, t_col]);
                        b_vec.push(current_weights[asset]);

                        // t_1 ≥ -(w_1 - w_current) → w_1 + t_1 ≥ w_current
                        a_data.extend([-1.0, -1.0]);
                        a_row_indices.extend([b_vec.len(), b_vec.len()]);
                        a_col_indices.extend([w_col, t_col]);
                        b_vec.push(-current_weights[asset]);
                    } else {
                        let prev_w_col = (step - 1) * self.variable_cnt + asset;
                        // For subsequent steps: t_k ≥ |w_k - w_{k-1}|

                        // t_k ≥ w_k - w_{k-1} → -w_k + w_{k-1} + t_k ≥ 0
                        a_data.extend([1.0, -1.0, -1.0]);
                        a_row_indices.extend([b_vec.len(), b_vec.len(), b_vec.len()]);
                        a_col_indices.extend([w_col, prev_w_col, t_col]);
                        b_vec.push(0.0);

                        // t_k ≥ -(w_k - w_{k-1}) → w_k - w_{k-1} + t_k ≥ 0
                        a_data.extend([-1.0, 1.0, -1.0]);
                        a_row_indices.extend([b_vec.len(), b_vec.len(), b_vec.len()]);
                        a_col_indices.extend([w_col, prev_w_col, t_col]);
                        b_vec.push(0.0);
                    }
                }
                cones.push(SupportedConeT::NonnegativeConeT(n * 2));
            }
        }

        let a_matrix = CscMatrix::new_from_triplets(
            b_vec.len(),           // Rows: budget + non-neg + transaction
            self.variable_cnt * t, // Columns: w + t
            a_row_indices,
            a_col_indices,
            a_data,
        );

        (a_matrix, b_vec, cones)
    }
}
