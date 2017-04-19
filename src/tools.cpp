#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

double Tools::NormalizeAngle(double angle)
{
	return fmod(angle+M_PI,2*M_PI) - M_PI;
}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
	VectorXd rmse(4);
	rmse << 0,0,0,0;

	// check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size
	if (estimations.size()==0)
	{
		std::cout << "CalculateRMSE: No measurements given" << std::endl;
		return rmse;
	}
	if (estimations.size()!=ground_truth.size())
	{
		std::cout << "CalculateRMSE: Measurements and ground_truth not same size" << std::endl;
		return rmse;
	}

	//accumulate squared residuals
	for(int i=0; i < estimations.size(); ++i){
		VectorXd res = estimations[i]-ground_truth[i];
		rmse += (res.array()*res.array()).matrix();
	}

	//calculate the mean
	rmse /= estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt().matrix();

	//return the result
	return rmse;
}
