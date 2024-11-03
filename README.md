# Static Dimension Kalman Filter

Static Dimension Kalman Filter implemented in Rust. The only dependency is [nalgebra](https://github.com/dimforge/nalgebra).

## Purpose

There are more than enough Kalman Filter implementations. So why would reinvent wheel?

- **It is simple and stupid**. Though many ones have tried, there is no "plug-and-play" Kalman Filter implementation. The real world use case is more complex and you always need to modify the original implementation before use. So why bother creating a "versatile" repository while in fact it is not? This repository is a merely template implementation for the simplest case, modify it based on your own use case.

- **It is a good learning material**. If you are learning Kalman Filter, it is a good start point. The core implementation is less than 20 lines of code, just read the code along with the head-scratching formulas. You can also check the unit test and understand what is going on in each iteration.

- **It exposes the prediction**. In many other implementations, predictions are override with posterior estimations. This, however, makes code hard to understand and limits some scenarios where prediction errors are important.

- **It is static dimension, so it is fast**. In most use cases, you never change the dimensions. So you don't need to use dynamic matrix and vector, since it involves heap allocation and is slower.

- **It is static dimension, so it is safe**. With static dimension implementation, it can check your input dimensions in compile time.

## Usage

You can refer to the unit test, but I'll paste it here as a minimal example:

```rust
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
```

The example exhibits how to track a simple speed-position system with Kalman Filter.

## A Little Bit of Theory

This repository faithfully implemented the vanilla Kalman Filter, without considering the control input $u_k$.

### Kalman Filter

The system is defined as:

$$\begin{aligned}
x_{k+1} &= F x_k + w_k \\
z_k &= H x_k + v_k
\end{aligned}$$

Where $F$ is state transition matrix and $H$ is observation matrix. $w_k$ is process noise with covariance $R$, $v_k$ is observation noise with covariance $Q$.

#### Time Update (Predict)

Project the state ahead:

$$ \hat{x}_{k+1}^- = F \hat{x}_k $$

Project the state error covariance ahead:

$$ P_{k+1}^- = F P_k F^T + Q $$

#### Measurement Update (Correct)

Compute prediction error covariance:

$$ S_k = H P_k^- H^T + R $$

Compute Kalman gain:

$$ K_k = P_k^- H^T S_k^{-1} $$

Update estimated state with new measurement:

$$ \hat{x}_k = \hat{x}_k^- + K_k(z_k - H\hat{x}_k^-) $$

Update the state error covariance:

$$ P_k = (I - K_k H) P_k^- $$


### My Comments

I don't want to waste too much time explain the formulas, you can find better explanation elsewhere. I only want to share some of my understandings about Kalman Filter.

- Kalman Filter's goal is to obtain an **optimal state estimation** ($\hat{x}_k$), given observation $z_k$ and a model $F$, both the observation and model have uncertainties. We can't trust either so we combine them with certain assumptions on the noise distribution.

- To combine the model prediction and observation, it simply use a weighted sum of those two. The **optimal weight** is the so-called `Kalman Gain`. The "optimal" is defined as the **minimization of posterior state error covariance**.

- In practical use cases, the difficulty is design of the model, and the determination of $Q$ and $R$, initial state does not really matter since it will converge. Moreover, the $F$, $H$, $Q$, $R$ can all be time-varying, and the numeric stability is not guaranteed. These problems bring extra difficulties in practical systems.

- Understand the idea behind the formulas is super important. In practice, you will encounter problems like "why doesn't my Kalman Filter converge?", "how do I know which observations are abnormal?". The answers can be found when deriving the formulas from scratch.
