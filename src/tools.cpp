#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  unsigned long number_of_estimations = estimations.size();

  if(number_of_estimations != ground_truth.size()
      || number_of_estimations == 0){
    std::cout << "Invalid estimation or ground_truth data" << std::endl;
    return rmse;
  }

  //accumulate squared residuals
  for (unsigned int i = 0; i < number_of_estimations; ++i) {

    VectorXd residual = estimations[i] - ground_truth[i];

    //coefficient-wise multiplication
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  //calculate the mean
  rmse = rmse / number_of_estimations;

  //calculate the squared root
  rmse = rmse.array().sqrt();

  //return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);

  // Extract state parameters
  double px = x_state(0);
  double py = x_state(1);
  const double vx = x_state(2);
  const double vy = x_state(3);

  // If px and py are very close to zero, we might end up in trouble, so we set them to small values.


  double px2_plus_py2 = px * px + py * py;

  if (fabs(px2_plus_py2) < 0.000001) {
    px2_plus_py2 = 0.0001;
  }

  double sqrt_px2_plus_py2 = sqrt(px2_plus_py2);

  // Range derivatives
  double derivative_range_respect_px = px / sqrt_px2_plus_py2;
  double derivative_range_respect_py = py / sqrt_px2_plus_py2;
  double derivative_range_respect_vx = 0;
  double derivative_range_respect_vy = 0;

  // Bearing derivatives
  double derivative_bearing_respect_px = -py / px2_plus_py2;
  double derivative_bearing_respect_py =  px / px2_plus_py2;
  double derivative_bearing_respect_vx = 0;
  double derivative_bearing_respect_vy = 0;

  // Range rate derivatives
  double derivative_range_rate_respect_px = py * (vx * py - vy * px) / (sqrt_px2_plus_py2 * px2_plus_py2);
  double derivative_range_rate_respect_py = px * (vy * px - vx * py) / (sqrt_px2_plus_py2 * px2_plus_py2);
  double derivative_range_rate_respect_vx = px / sqrt_px2_plus_py2;
  double derivative_range_rate_respect_vy = py / sqrt_px2_plus_py2;

  Hj << derivative_range_respect_px, derivative_range_respect_py, derivative_range_respect_vx, derivative_range_respect_vy,
        derivative_bearing_respect_px, derivative_bearing_respect_py, derivative_bearing_respect_vx, derivative_bearing_respect_vy,
        derivative_range_rate_respect_px, derivative_range_rate_respect_py, derivative_range_rate_respect_vx, derivative_range_rate_respect_vy;

  return Hj;
}
