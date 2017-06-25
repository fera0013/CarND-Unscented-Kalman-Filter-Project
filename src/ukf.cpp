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

	// Initially set to false, set to true in first call of ProcessMeasurement
	is_initialized_ = false;

	// if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;

	// if this is false, radar measurements will be ignored (except during init)
	use_radar_ = true;

	// time when the state is true, in us
	time_us_ = 0;

	// Process noise standard deviation longitudinal acceleration in m/s^2
	std_a_ = 1.0;      

	// Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = 1.0;   

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

	// the current NIS for radar
	NIS_radar_ = 0.0;

	// the current NIS for laser
	NIS_laser_ = 0.0;

	// State dimension
	n_x_ = 5;

	// Augmented state dimension
	n_aug_ = 7;

	// initial state vector
	x_ = VectorXd(n_x_);

	// initial covariance matrix
	P_ = MatrixXd(n_x_, n_x_);

	// predicted sigma points matrix
	Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

	// Sigma point spreading parameter
	lambda_ = 3.0 - n_aug_;

	// Weights of sigma points
	weights_ = VectorXd(2 * n_aug_ + 1);
	weights_(0) = lambda_ / (lambda_ + n_aug_);
	for (int i = 1; i<2 * n_aug_ + 1; i++) {
		weights_(i) = 0.5 / (n_aug_ + lambda_);
	}
}

UKF::~UKF() {}

/**
* @param {MeasurementPackage} meas_package The latest measurement data of
* either radar or laser.
*/
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
	if (!is_initialized_) 
	{ 
		x_ << 0.0, 0.0, 3.0, 0.0, 0.1;       
		P_ = MatrixXd::Identity(n_x_, n_x_); 
		P_(2, 2) = 1.0*1.0;
		P_(3, 3) = M_PI*M_PI / 64.0;
		P_(4, 4) = P_(3, 3) / 10.0;
		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			float rho = meas_package.raw_measurements_[1]; 
			x_(0) = meas_package.raw_measurements_[0] * cos(rho);
			x_(1) = meas_package.raw_measurements_[0] * sin(rho);
			P_(0, 0) = std_radr_*std_radr_*0.5;
			P_(1, 1) = std_radr_*std_radr_*0.5;
			NIS_radar_ = 0.0;
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			x_(0) = meas_package.raw_measurements_[0];
			x_(1) = meas_package.raw_measurements_[1];
			P_(0, 0) = std_laspx_*std_laspx_;
			P_(1, 1) = std_laspy_*std_laspy_;
			NIS_laser_ = 0.0;
		}
		time_us_ = meas_package.timestamp_;

		// done initializing, no need to predict or update
		is_initialized_ = true;
		return;
	}
	// Calculate time delta to last update
	double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
	time_us_ = meas_package.timestamp_;

	Prediction(dt);

	if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		UpdateRadar(meas_package);
	}
	else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
		UpdateLidar(meas_package);
	}
}

/**
* Predicts sigma points, the state, and the state covariance matrix.
* @param {double} delta_t the change in time (in seconds) between the last
* measurement and this one.
*/
void UKF::Prediction(double delta_t) {
	//create augmented mean vector
	VectorXd x_aug = VectorXd(n_aug_);

	//create augmented state covariance
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

	//create sigma point matrix
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

	//create augmented mean state
	x_aug.head(n_x_) = x_;
	x_aug(n_x_) = 0;
	x_aug(n_x_ + 1) = 0;

	//create augmented covariance matrix
	P_aug.fill(0.0);
	P_aug.topLeftCorner(n_x_, n_x_) = P_;
	P_aug(n_x_, n_x_) = std_a_*std_a_;
	P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_*std_yawdd_;

	//create square root matrix
	MatrixXd L = P_aug.llt().matrixL();

	//create augmented sigma points
	double sqrt_lam_aug = sqrt(lambda_ + n_aug_);
	Xsig_aug.col(0) = x_aug;
	for (int i = 0; i< n_aug_; i++) {
		Xsig_aug.col(i + 1) = x_aug + sqrt_lam_aug * L.col(i);
		Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt_lam_aug * L.col(i);
	}

	for (int i = 0; i< 2 * n_aug_ + 1; i++) {
		//extract values for better readability
		double p_x = Xsig_aug(0, i);
		double p_y = Xsig_aug(1, i);
		double v = Xsig_aug(2, i);
		double yaw = Xsig_aug(3, i);
		double yawd = Xsig_aug(4, i);
		double nu_a = Xsig_aug(5, i);
		double nu_yawdd = Xsig_aug(6, i);

		//predicted state values
		double px_p, py_p;

		//avoid division by zero
		if (fabs(yawd) > 0.001) {
			px_p = p_x + v / yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
			py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd*delta_t));
		}
		else {
			px_p = p_x + v*delta_t*cos(yaw);
			py_p = p_y + v*delta_t*sin(yaw);
		}

		double v_p = v;
		double yaw_p = yaw + yawd*delta_t;
		double yawd_p = yawd;

		//add noise
		px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
		py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
		v_p = v_p + nu_a*delta_t;

		yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
		yawd_p = yawd_p + nu_yawdd*delta_t;

		//write predicted sigma point into right column
		Xsig_pred_(0, i) = px_p;
		Xsig_pred_(1, i) = py_p;
		Xsig_pred_(2, i) = v_p;
		Xsig_pred_(3, i) = yaw_p;
		Xsig_pred_(4, i) = yawd_p;
	}

    // Predict state mean
	x_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
		x_ = x_ + weights_(i) * Xsig_pred_.col(i);
	}

	// Predict state covariance matrix
	P_.fill(0.0);
	VectorXd x_diff = VectorXd(n_x_);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

												// state difference
		x_diff = Xsig_pred_.col(i) - x_;
		//angle normalization
		x_diff(3) = fmod(x_diff(3) + M_PI, (2.0*M_PI)) - M_PI;

		P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
	}
}

/**
* Updates the state and the state covariance matrix using a laser measurement.
* @param {MeasurementPackage} meas_package
*/
void UKF::UpdateLidar(MeasurementPackage meas_package) {
	if (!use_laser_) {
		NIS_laser_ = 0.0;
		return;
	}
	int n_z = 2; 
	MatrixXd H = MatrixXd(n_z, n_x_);
	H.fill(0.0);
	H(0, 0) = 1.0;
	H(1, 1) = 1.0;

	MatrixXd R = MatrixXd(n_z, n_z);
	R.fill(0.0);
	R(0, 0) = std_laspx_*std_laspx_;
	R(1, 1) = std_laspy_*std_laspy_;

	MatrixXd HP = MatrixXd(n_z, n_x_);
	HP = H * P_;

	VectorXd z_diff = meas_package.raw_measurements_ - H * x_;
	MatrixXd Ht = H.transpose();
	MatrixXd S = HP * Ht + R;
	MatrixXd Si = S.inverse();
	MatrixXd K = HP.transpose() * Si;

	x_ = x_ + (K * z_diff);
	P_ = P_ - K * HP;

	NIS_laser_ = z_diff.transpose() * Si * z_diff;
}

/**
* Updates the state and the state covariance matrix using a radar measurement.
* @param {MeasurementPackage} meas_package
*/
void UKF::UpdateRadar(MeasurementPackage meas_package) {
	if (!use_radar_) {
		NIS_radar_ = 0.0;
		return;
	}
	int n_z = 3; 

	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  
		double p_x = Xsig_pred_(0, i);
		double p_y = Xsig_pred_(1, i);
		double v = Xsig_pred_(2, i);
		double yaw = Xsig_pred_(3, i);

		double v1 = cos(yaw)*v;
		double v2 = sin(yaw)*v;

		Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y);     

		//Avoid too small numbers
		if (Zsig(0, i)< 0.001) {                            
			Zsig(0, i) = 0.001;
			Zsig(1, i) = 0.0;
		}
		else {
			Zsig(1, i) = atan2(p_y, p_x);               
		}
		Zsig(2, i) = (p_x*v1 + p_y*v2) / Zsig(0, i);  
	}

	// Predicted measurement mean
	VectorXd z_pred = VectorXd(n_z);
	z_pred.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	// Measurement covariance matrix
	MatrixXd S = MatrixXd(n_z, n_z);
	S.fill(0.0);
	VectorXd z_diff = VectorXd(n_z);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  
		z_diff = Zsig.col(i) - z_pred;
		//angle normalization
		z_diff(1) = fmod(z_diff(1) + M_PI, (2.0*M_PI)) - M_PI;

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}
	// add measurement noise covariance matrix
	S(0, 0) += std_radr_*std_radr_;
	S(1, 1) += std_radphi_*std_radphi_;
	S(2, 2) += std_radrd_*std_radrd_;

	MatrixXd Tc = MatrixXd(n_x_, n_z);
	Tc.fill(0.0);
	VectorXd x_diff = VectorXd(n_x_);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) { 
		z_diff = Zsig.col(i) - z_pred;
		//angle normalization
		z_diff(1) = fmod(z_diff(1) + M_PI, (2.0*M_PI)) - M_PI;

		// state difference
		x_diff = Xsig_pred_.col(i) - x_;
		//angle normalization
		x_diff(3) = fmod(x_diff(3) + M_PI, (2.0*M_PI)) - M_PI;

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

	// Kalman gain K;
	MatrixXd Si = MatrixXd(n_z, n_z);
	Si = S.inverse();
	MatrixXd K = Tc * Si;

	//residual
	z_diff = meas_package.raw_measurements_ - z_pred;

	//angle normalization
	z_diff(1) = fmod(z_diff(1) + M_PI, (2.0*M_PI)) - M_PI;

	// Update state mean and covariance matrix
	x_ = x_ + K * z_diff;
	P_ = P_ - Tc*K.transpose(); 

	NIS_radar_ = z_diff.transpose() * Si * z_diff;
}