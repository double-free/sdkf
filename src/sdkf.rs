use nalgebra as na;

pub struct KalmanFilter<T: na::RealField, const STATE_DIM: usize, const MEASURE_DIM: usize> {
    // predicted state
    x: na::SVector<T, STATE_DIM>,
    pred_x: na::SVector<T, STATE_DIM>,

    // prediction error covariance
    pred_err_cov: na::SMatrix<T, MEASURE_DIM, MEASURE_DIM>,

    // state error covariance
    p_mat: na::SMatrix<T, STATE_DIM, STATE_DIM>,
    pred_p_mat: na::SMatrix<T, STATE_DIM, STATE_DIM>,

    // state transition matrix
    f_mat: na::SMatrix<T, STATE_DIM, STATE_DIM>,
    // observation matrix
    h_mat: na::SMatrix<T, MEASURE_DIM, STATE_DIM>,
    // Process noise
    q_mat: na::SMatrix<T, STATE_DIM, STATE_DIM>,
    // Measurement noise
    r_mat: na::SMatrix<T, MEASURE_DIM, MEASURE_DIM>,
}

impl<T: na::RealField, const STATE_DIM: usize, const MEASURE_DIM: usize>
    KalmanFilter<T, STATE_DIM, MEASURE_DIM>
{
    pub fn new(
        init_x: &na::SVector<T, STATE_DIM>,
        init_p_mat: &na::SMatrix<T, STATE_DIM, STATE_DIM>,
        f_mat: &na::SMatrix<T, STATE_DIM, STATE_DIM>,
        h_mat: &na::SMatrix<T, MEASURE_DIM, STATE_DIM>,
        q_mat: &na::SMatrix<T, STATE_DIM, STATE_DIM>,
        r_mat: &na::SMatrix<T, MEASURE_DIM, MEASURE_DIM>,
    ) -> Self {
        let pred_x = f_mat * init_x;
        let pred_p_mat = f_mat * init_p_mat * f_mat.transpose() + q_mat;
        let pred_err_cov = h_mat * &pred_p_mat * h_mat.transpose() + r_mat;
        KalmanFilter {
            x: init_x.clone(),
            pred_x: pred_x,
            pred_err_cov: pred_err_cov,
            p_mat: init_p_mat.clone(),
            pred_p_mat: pred_p_mat,
            f_mat: f_mat.clone(),
            h_mat: h_mat.clone(),
            q_mat: q_mat.clone(),
            r_mat: r_mat.clone(),
        }
    }

    // update state with new observation, return the prediction error
    // NOTE: we assume that F, H, R, Q are constant, which is not true in some scenarios
    pub fn update(
        &mut self,
        measure: &na::SVector<T, MEASURE_DIM>,
    ) -> Result<na::SVector<T, MEASURE_DIM>, String> {
        // prediction
        self.pred_x = &self.f_mat * &self.x;
        self.pred_p_mat = &self.f_mat * &self.p_mat * self.f_mat.transpose() + &self.q_mat;

        // update with innovation
        self.pred_err_cov = &self.h_mat * &self.pred_p_mat * &self.h_mat.transpose() + &self.r_mat;
        let result = match self.pred_err_cov.clone().try_inverse() {
            None => Err(format!("can not inverse {:?}", self.pred_err_cov)),
            Some(inversed) => {
                let kalman_gain = &self.pred_p_mat * self.h_mat.transpose() * inversed;
                let pred_err = measure - &self.h_mat * &self.pred_x;
                self.x = &self.pred_x + &kalman_gain * &pred_err;
                let eye = na::SMatrix::identity();
                self.p_mat = (eye - &kalman_gain * &self.h_mat) * &self.pred_p_mat;
                return Ok(pred_err);
            }
        };
        return result;
    }

    pub fn state(&self) -> &na::SVector<T, STATE_DIM> {
        return &self.x;
    }
}
