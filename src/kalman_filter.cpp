#include <iostream>
#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd F_transpose = F_.transpose();

  P_ = F_ * P_ * F_transpose + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  MatrixXd P_by_H_transpose = P_ * H_.transpose();

  VectorXd z_predicted = H_ * x_;
  VectorXd y = z - z_predicted;
  MatrixXd S = H_ * P_by_H_transpose + R_;
  MatrixXd S_inverse = S.inverse();
  MatrixXd K = P_by_H_transpose * S_inverse;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  double px = x_(0);
  double py = x_(1);
  const double vx = x_(2);
  const double vy = x_(3);

  VectorXd z_predicted(3);
  const double square_root_px2_plus_py2 = sqrt(px * px + py * py);
  double range = square_root_px2_plus_py2;
  double bearing = atan2(py, px);
  double range_rate = (px * vx + py * vy) / square_root_px2_plus_py2;

  while (bearing > M_PI || bearing < -M_PI) {
    if (bearing > M_PI) {
      bearing -= 2 * M_PI;
    } else {
      bearing += 2 * M_PI;
    }
  }

  z_predicted << range,
                 bearing,
                 range_rate;

  VectorXd y = z - z_predicted;
  MatrixXd H_transpose = H_.transpose();
  MatrixXd P_by_H_transpose = P_ * H_transpose;;
  MatrixXd S = H_ * P_by_H_transpose + R_;
  MatrixXd S_inverse = S.inverse();
  MatrixXd K = P_by_H_transpose * S_inverse;


  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
