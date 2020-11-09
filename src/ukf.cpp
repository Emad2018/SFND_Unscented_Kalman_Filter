#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values
   */

  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 + n_aug_;

  // initial state vector
  x_ = VectorXd(n_x_);
  x_.fill(0);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  P_.fill(0);
  P_ << std_radr_ * std_radr_, 0, 0, 0, 0, 0, std_radr_ * std_radr_, 0, 0, 0, 0,
      0, std_radrd_ * std_radrd_, 0, 0, 0, 0, 0, std_radphi_, 0, 0, 0, 0, 0,
      std_radphi_;
  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred_.fill(0);

  // Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_.fill(0);
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.5;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  // Initialization//

  // skip processing if the both sensors are ignored
  if ((meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) ||
      (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)) {

    if (!is_initialized_) {
      if (meas_package.sensor_type_ == MeasurementPackage::LASER &&
          use_laser_) {
        // Initialize state.
        x_(0) = meas_package.raw_measurements_(0);
        x_(1) = meas_package.raw_measurements_(1);

      } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR &&
                 use_radar_) {
        // Convert polar to cartesian coordinates
        float rho = meas_package.raw_measurements_(0);
        float theta = meas_package.raw_measurements_(1);
        float rho_dot = meas_package.raw_measurements_(2);

        // initialize state
        x_(0) = rho * cos(theta);
        x_(1) = rho * sin(theta);
        x_(2) = rho * rho_dot;
        x_(3) = theta;
      }

      time_us_ = meas_package.timestamp_;

      is_initialized_ = true;

      return;
    }

    // Prediction//

    // compute the time elapsed between the current and previous measurements
    float dt = (meas_package.timestamp_ - time_us_) /
               1000000.0; // dt - expressed in seconds
    time_us_ = meas_package.timestamp_;
    Prediction(dt);

    // Update//

    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      UpdateLidar(meas_package);
    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      UpdateRadar(meas_package);
    }
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location.
   * Modify the state vector, x_. Predict sigma points, the state,
   * and the state covariance matrix.
   */

  //******************************************************
  // Generate Sigma points
  //******************************************************

  // create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.fill(0);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.fill(0);

  // Fill augmented mean vector
  x_aug.head(n_x_) = x_;

  // Fill augmented covariance matrix
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  // create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug.col(0) = x_aug;

  for (int index = 0; index < n_aug_; ++index) {
    Xsig_aug.col(index + 1) = x_aug + (sqrt(lambda_ + n_aug_) * L.col(index));
    Xsig_aug.col(index + 1 + n_aug_) =
        x_aug - (sqrt(lambda_ + n_aug_) * L.col(index));
  }
  //******************************************************
  // Generate Sigma points END
  //******************************************************

  //******************************************************
  // Predict Sigma points
  //******************************************************
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // extract values for better readability
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    } else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    // write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }
  //******************************************************
  // Predict Sigma points  END
  //******************************************************

  //******************************************************
  // Predict Mean and Coveriance
  //******************************************************

  // create vector for predicted state
  VectorXd x_pred = VectorXd(n_x_);
  x_pred.fill(0.0);

  // create covariance matrix for prediction
  MatrixXd P_pred = MatrixXd(n_x_, n_x_);
  P_pred.fill(0.0);

  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {
    weights_(i) = .5 / (lambda_ + n_aug_);
  }
  // predict state mean
  x_pred = Xsig_pred_ * weights_;
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) { // iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_pred;
    // angle normalization
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    P_pred = P_pred + weights_(i) * x_diff * x_diff.transpose();
  }
  x_ = x_pred;
  P_ = P_pred;
  //******************************************************
  // Predict Mean and Coveriance END
  //******************************************************
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  /*
 // Radar Sensor dimension
 int n_z_ = 3;

 // Measurments Sigma points
 MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);
 Zsig.fill(0);

 // Sensor Measurments
 VectorXd z_measurment = VectorXd(n_z_);
 z_measurment.fill(0);
 z_measurment << meas_package.raw_measurements_(0),
     meas_package.raw_measurements_(1), meas_package.raw_measurements_(2);

 // mean predicted measurement
 VectorXd z_pred = VectorXd(n_z_);
 z_pred.fill(0);

 // measurement covariance matrix S
 MatrixXd S = MatrixXd(n_z_, n_z_);
 S.fill(0);

 // transform sigma points into measurement space
 for (int i = 0; i < 2 * n_aug_ + 1; i++) {
   float rho = sqrt(Xsig_pred_(0, i) * Xsig_pred_(0, i) +
                    Xsig_pred_(1, i) * Xsig_pred_(1, i));

   float theta = atan2(Xsig_pred_(1, i), Xsig_pred_(0, i));
   float rho_rate =
       (Xsig_pred_(0, i) * cos(Xsig_pred_(3, i)) * Xsig_pred_(2, i) +
        Xsig_pred_(1, i) * sin(Xsig_pred_(3, i)) * Xsig_pred_(2, i)) /
       rho;
   Zsig.col(i) << rho, theta, rho_rate;
 }
 // calculate mean predicted measurement
 z_pred = Zsig * weights_;
 // calculate innovation covariance matrix S
 MatrixXd R = MatrixXd(n_z_, n_z_);
 R.fill(0);
 R(0, 0) = std_radr_ * std_radr_;
 R(1, 1) = std_radphi_ * std_radphi_;
 R(2, 2) = std_radrd_ * std_radrd_;

 for (int i = 0; i < 2 * n_aug_ + 1; ++i) { // 2n+1 simga points
   // residual
   VectorXd z_diff = Zsig.col(i) - z_pred;

   // angle normalization
   while (z_diff(1) > M_PI)
     z_diff(1) -= 2. * M_PI;
   while (z_diff(1) < -M_PI)
     z_diff(1) += 2. * M_PI;

   S = S + weights_(i) * z_diff * z_diff.transpose();
 }

 S += R;

 //*************************************************************
 // UKF Update
 //************************************************************
 MatrixXd Tc = MatrixXd(n_x_, n_z_);
 Tc.fill(0.0);
 for (int i = 0; i < 2 * n_aug_ + 1; ++i) { // 2n+1 simga points
   // residual
   VectorXd z_diff = Zsig.col(i) - z_pred;
   // angle normalization
   while (z_diff(1) > M_PI)
     z_diff(1) -= 2. * M_PI;
   while (z_diff(1) < -M_PI)
     z_diff(1) += 2. * M_PI;

   // state difference
   VectorXd x_diff = Xsig_pred_.col(i) - x_;
   // angle normalization
   while (x_diff(3) > M_PI)
     x_diff(3) -= 2. * M_PI;
   while (x_diff(3) < -M_PI)
     x_diff(3) += 2. * M_PI;

   Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
 }

 // Kalman gain K;
 MatrixXd K = Tc * S.inverse();

 // residual

 VectorXd z_diff = z_measurment - z_pred;

 // angle normalization
 while (z_diff(1) > M_PI)
   z_diff(1) -= 2. * M_PI;
 while (z_diff(1) < -M_PI)
   z_diff(1) += 2. * M_PI;

 // update state mean and covariance matrix
 x_ = x_ + K * z_diff;
 P_ = P_ - K * S * K.transpose();
 */
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  // Radar Sensor dimension
  int n_z_ = 3;

  // Measurments Sigma points
  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);
  Zsig.fill(0);

  // Sensor Measurments
  VectorXd z_measurment = VectorXd(n_z_);
  z_measurment.fill(0);
  z_measurment << meas_package.raw_measurements_(0),
      meas_package.raw_measurements_(1), meas_package.raw_measurements_(2);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);
  z_pred.fill(0);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_, n_z_);
  S.fill(0);

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    float rho = sqrt(Xsig_pred_(0, i) * Xsig_pred_(0, i) +
                     Xsig_pred_(1, i) * Xsig_pred_(1, i));

    float theta = atan2(Xsig_pred_(1, i), Xsig_pred_(0, i));
    float rho_rate =
        (Xsig_pred_(0, i) * cos(Xsig_pred_(3, i)) * Xsig_pred_(2, i) +
         Xsig_pred_(1, i) * sin(Xsig_pred_(3, i)) * Xsig_pred_(2, i)) /
        rho;
    Zsig.col(i) << rho, theta, rho_rate;
  }
  // calculate mean predicted measurement
  z_pred = Zsig * weights_;
  // calculate innovation covariance matrix S
  MatrixXd R = MatrixXd(n_z_, n_z_);
  R.fill(0);
  R(0, 0) = std_radr_ * std_radr_;
  R(1, 1) = std_radphi_ * std_radphi_;
  R(2, 2) = std_radrd_ * std_radrd_;

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) { // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  S += R;

  //*************************************************************
  // UKF Update
  //************************************************************
  MatrixXd Tc = MatrixXd(n_x_, n_z_);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) { // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual

  VectorXd z_diff = z_measurment - z_pred;

  // angle normalization
  while (z_diff(1) > M_PI)
    z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI)
    z_diff(1) += 2. * M_PI;

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}