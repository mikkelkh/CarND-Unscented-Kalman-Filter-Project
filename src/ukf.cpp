#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
	// if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;

	// if this is false, radar measurements will be ignored (except during init)
	use_radar_ = true;

	// initial state vector
	x_ = VectorXd(5);

	// initial covariance matrix
	P_ = MatrixXd::Zero(5, 5);

	// Process noise standard deviation longitudinal acceleration in m/s^2
	std_a_ = 3;//30;

	// Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = 0.6*M_PI;//30;

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
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
	 */

	is_initialized_ = false;

	// Initialize state dimensions
	n_x_ = 5;
	n_aug_ = 7;

	// Choose lambda for how Sigma points are distributed
	lambda_ = 3-n_aug_;

	// Set weights
	weights_ = VectorXd(2*n_aug_+1);
	weights_(0) = lambda_/(lambda_+n_aug_);
	double weight = 1/(2*(lambda_+n_aug_));
	for (int i=1;i<2*n_aug_+1;i++)
	{
		weights_(i) = weight;
	}

	// Initialize process covariance matrix.
	// Values are inspired from udacity discussion forum.
	P_ = MatrixXd::Zero(n_x_, n_x_);
	P_(0,0) = 2;
	P_(1,1) = 4;
	P_(2,2) = 1;
	P_(3,3) = 0.5;
	P_(4,4) = 0.5;

	// Initialize laser measurement update function, which is linear.
	H_laser_ = MatrixXd::Zero(2,n_x_);
	H_laser_ << 1,0,0,0,0,
			    0,1,0,0,0;

	// Initialize laser measurement noise covariance matrix.
	R_laser_ = MatrixXd::Zero(2,2);
	R_laser_(0,0) = std_laspx_*std_laspx_;
	R_laser_(1,1) = std_laspy_*std_laspy_;

	// Initialize radar measurement noise covariance matrix
	R_radar_ = MatrixXd::Zero(3,3);
	R_radar_(0,0) = std_radr_*std_radr_;
	R_radar_(1,1) = std_radphi_*std_radphi_;
	R_radar_(2,2) = std_radrd_*std_radrd_;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
	/**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
	 */

	// Return if the current measurement is disabled (laser or radar)
	if ((meas_package.sensor_type_ == MeasurementPackage::RADAR) && !use_radar_)
		return;

	if ((meas_package.sensor_type_ == MeasurementPackage::LASER) && !use_laser_)
		return;

	/*****************************************************************************
	 *  Initialization
	 ****************************************************************************/
	if (!is_initialized_) {
		/**
	TODO:
		 * Initialize the state ekf_.x_ with the first measurement.
		 * Create the covariance matrix.
		 * Remember: you'll need to convert radar from polar to cartesian coordinates.
		 */
		cout << "UKF: " << endl;

		// Update time
		time_us_ = meas_package.timestamp_;
		x_ = VectorXd::Zero(5);

		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			/**
	  	  	  Convert radar from polar to cartesian coordinates and initialize state.
			 */
			double range = meas_package.raw_measurements_[0];
			double bearing = meas_package.raw_measurements_[1];
			//		double velocity = meas_package.raw_measurements_[2];

			x_(0) = range*cos(bearing);
			x_(1) = range*sin(bearing);
			x_(2) = 0; // No information available
			x_(3) = 0; // No information available
			x_(4) = 0; // No information available
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			/**
	  	  	  Initialize state.
			 */
			x_(0) = meas_package.raw_measurements_[0];
			x_(1) = meas_package.raw_measurements_[1];
			x_(2) = 0; // No information available
			x_(3) = 0; // No information available
			x_(4) = 0; // No information available
		}

		// Done initializing, no need to predict or update
		is_initialized_ = true;
		return;
	}

	/*****************************************************************************
	 *  Prediction
	 ****************************************************************************/
	double delta_t = (meas_package.timestamp_-time_us_)/1000000.0;
	time_us_ = meas_package.timestamp_;

	// Handle numerical instability by subdividing the prediction steps into incremental updates.
	// Idea from Wolfgang_Steiner at udacity discussion forum: https://discussions.udacity.com/t/numerical-instability-of-the-implementation/230449/2
	while (delta_t > 0.1)
	{
		const double dt = 0.05;
		Prediction(dt);
		delta_t -= dt;
	}
	Prediction(delta_t);

	/*****************************************************************************
	 *  Update
	 ****************************************************************************/

	if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		// Radar updates
		UpdateRadar(meas_package);
	} else {
		// Laser updates
		UpdateLidar(meas_package);
	}

	// Print the output
	cout << "x_ = " << x_ << endl;
	cout << "P_ = " << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
	/**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
	 */

	/* First, we calculate Sigma points and augmented covariance matrix */
	VectorXd x_aug = VectorXd::Zero(n_aug_);
	x_aug.head(n_x_) = x_;

	// Create augmented covariance matrix
	MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
	P_aug.topLeftCorner(n_x_, n_x_) = P_;
	P_aug(n_aug_-2,n_aug_-2) = std_a_*std_a_;
	P_aug(n_aug_-1,n_aug_-1) = std_yawdd_*std_yawdd_;

	// Create square roots
	MatrixXd A_aug = P_aug.llt().matrixL();
	double sqrt_lambda_n_aug = sqrt(lambda_+n_aug_);

	// Create augmented sigma points
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
	Xsig_aug.col(0) = x_aug;
	for (int i = 0; i < n_aug_; i++)
	{
		Xsig_aug.col(i+1)     = x_aug + sqrt_lambda_n_aug * A_aug.col(i);
		Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt_lambda_n_aug * A_aug.col(i);
	}

	/* Then, we predict all the Sigma points */
	// Predict sigma points
	Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);
	for (int i=0;i<Xsig_aug.cols();i++)
	{
		double px = Xsig_aug(0,i);
		double py = Xsig_aug(1,i);
		double v = Xsig_aug(2,i);
		double psi = Xsig_aug(3,i);
		double psi_dot = Xsig_aug(4,i);
		double na = Xsig_aug(5,i);
		double n_psi_dot2 = Xsig_aug(6,i);

		if (fabs(psi_dot) <= 0.001)
		{
			Xsig_pred_(0,i) = px      + v*cos(psi)*delta_t                                + 0.5*delta_t*delta_t*cos(psi)*na;
			Xsig_pred_(1,i) = py      + v*sin(psi)*delta_t                                + 0.5*delta_t*delta_t*sin(psi)*na;
			Xsig_pred_(2,i) = v                                                           + delta_t*na;
			Xsig_pred_(3,i) = psi                                                         + 0.5f*delta_t*delta_t*n_psi_dot2;
			Xsig_pred_(4,i) = psi_dot                                                     + delta_t*n_psi_dot2;
		}
		else
		{
			Xsig_pred_(0,i) = px      + v/psi_dot*(sin(psi+psi_dot*delta_t)-sin(psi))     + 0.5*delta_t*delta_t*cos(psi)*na;
			Xsig_pred_(1,i) = py      + v/psi_dot*(-cos(psi+psi_dot*delta_t)+cos(psi))    + 0.5*delta_t*delta_t*sin(psi)*na;
			Xsig_pred_(2,i) = v                                                           + delta_t*na;
			Xsig_pred_(3,i) = psi     + psi_dot*delta_t                                   + 0.5*delta_t*delta_t*n_psi_dot2;
			Xsig_pred_(4,i) = psi_dot                                                     + delta_t*n_psi_dot2;
		}
	}

	/* Based on the predicted Sigma points, we can calculate the new predicted mean and covariance */
	// Predict state mean
	x_ = Xsig_pred_ * weights_;

	// Predict state covariance matrix
	P_ = MatrixXd::Zero(n_x_, n_x_);
	for (int i=0;i<Xsig_pred_.cols();i++)
	{
		MatrixXd xdiff = (Xsig_pred_.col(i)-x_);
		xdiff(3) = tools_.NormalizeAngle(xdiff(3));
		P_ += weights_(i)*xdiff*xdiff.transpose();
	}
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
	/**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
	 */
	VectorXd z = meas_package.raw_measurements_;

	// Use traditional kalman update step, as the measurement function is linear
	VectorXd y = z-H_laser_*x_;
	MatrixXd S = H_laser_*P_*H_laser_.transpose()+R_laser_;
	MatrixXd K = P_*H_laser_.transpose()*S.inverse();
	MatrixXd I = MatrixXd::Identity(H_laser_.cols(),H_laser_.cols());

	// new state
	x_ = x_+K*y;
	P_ = (I-K*H_laser_)*P_;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
	/**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
	 */
	VectorXd z = meas_package.raw_measurements_;
	int n_z = 3;

	// Transform sigma points into measurement space
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
	for (int i=0;i<Xsig_pred_.cols();i++)
	{
		double px = Xsig_pred_(0,i);
		double py = Xsig_pred_(1,i);
		double v = Xsig_pred_(2,i);
		double psi = Xsig_pred_(3,i);

		double range = sqrt(px*px+py*py);

		Zsig(0,i) = range;
		Zsig(1,i) = atan2(py,px);

		// Handle numerical instability when denominator is small
		if (range > 0.001)
			Zsig(2,i) = (px*cos(psi)*v+py*sin(psi)*v)/range;
		else
			Zsig(2,i) = 0;
	}

	// Calculate mean predicted measurement
	VectorXd z_pred = VectorXd::Zero(n_z);
	z_pred = Zsig * weights_;

	// Calculate cross correlation matrix and measurement covariance matrix S
	MatrixXd Tc = MatrixXd::Zero(n_x_,n_z);
	MatrixXd S = MatrixXd::Zero(n_z,n_z);
	for (int i=0;i<Zsig.cols();i++)
	{
		VectorXd zdiff = Zsig.col(i) - z_pred;
		zdiff(1) = tools_.NormalizeAngle(zdiff(1));

		S += weights_(i)*zdiff*zdiff.transpose();

		VectorXd xdiff = Xsig_pred_.col(i) - x_;
		xdiff(3) = tools_.NormalizeAngle(xdiff(3));

		Tc += weights_(i)*xdiff*zdiff.transpose();
	}

	// Add measurement noise
	S += R_radar_;

	// Calculate Kalman gain K;
	MatrixXd Sinv = S.inverse();
	MatrixXd K = Tc*Sinv;

	// Update state mean and covariance matrix
	VectorXd z_diff = z-z_pred;
	z_diff(1) = tools_.NormalizeAngle(z_diff(1));
	x_ = x_+K*z_diff;
	x_(3) = tools_.NormalizeAngle(x_(3));
	P_ = P_-K*S*K.transpose();

	// Calculate NIS
	NIS_radar_ = z_diff.transpose()*Sinv*z_diff;
}
